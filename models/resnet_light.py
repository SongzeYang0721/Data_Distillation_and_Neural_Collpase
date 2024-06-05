import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

"""From https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18"""

__all__ = ['LightEncoder', 'resnet18_light', 'resnet20_light']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LightBasicBlockEnc(nn.Module):
    """The basic block architecture of resnet-18 network for smaller input images."""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        outdim: int = 0
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if outdim == 0:
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
        else:
            self.conv2 = conv3x3(planes, outdim)
            self.bn2 = norm_layer(outdim)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LightEncoder(nn.Module):
    """The encoder model, following the architecture of resnet-18 
    for smaller input images."""
    def __init__(
        self,
        block: Type[LightBasicBlockEnc],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fc_bias: bool = True,
        fixdim: int = False,
        ETF_fc: bool = False,
        SOTA: bool = False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.SOTA = SOTA
        if SOTA:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False) # does not change heights and wideth
        else:
            # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
            #                     bias=False) # height and width 32 ----> 16
            self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        
        if not fixdim:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.fc = nn.Linear(64*8*8 * block.expansion, num_classes, bias=fc_bias)
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                           outdim=num_classes)
            self.fc = nn.Linear(num_classes, num_classes, bias=fc_bias)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if ETF_fc:
                    weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
                    weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
                    if fixdim:
                        m.weight = nn.Parameter(weight)
                    else:
                        m.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, 64*8*8 * block.expansion)))
                    m.weight.requires_grad_(False)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LightBasicBlockEnc) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: Type[LightBasicBlockEnc],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    outdim: int = 0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride), 
                # If we use conv1x1 here, then we should also use it in the decoder 
                # part. But some pixels will be left unconstructed (simple noise) on decoding.
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, 
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        downsample = None
        if outdim != 0:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outdim),
                norm_layer(outdim),
            )

        layers.append(block(self.inplanes, 
                            planes, 
                            groups=self.groups,
                            base_width=self.base_width, 
                            dilation=self.dilation,
                            norm_layer=norm_layer, 
                            downsample=downsample, 
                            outdim=outdim))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = torch.flatten(x, 1) # flatten the second dimension from (n, m, k) to (n, m*k), here m*k = d
        features = x # normalized H ready to feed to the linear layer
        x = self.fc(x)


        return x, features 
         
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

def _resnet_light(
    arch: str,
    block: Type[Union[LightBasicBlockEnc]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> LightEncoder:
    model = LightEncoder(block, layers, **kwargs)
    if pretrained:
        sys.exit('No pre-trained model is allowed here!')
    return model

def resnet18_light(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LightEncoder:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_light('resnet18_light', LightBasicBlockEnc, [2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet20_light(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LightEncoder:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_light('resnet20_light', LightBasicBlockEnc, [3, 3, 3], pretrained, progress,
                   **kwargs)