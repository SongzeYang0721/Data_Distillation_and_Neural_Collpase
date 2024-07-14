import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_DECODER(nn.Module):
    def __init__(self, hidden, depth = 6, fc_bias = True, num_classes = 10):
        # Depth means how many layers before final linear layer
        
        super(MLP_DECODER, self).__init__()
        layers = [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(1, depth):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
            if i == depth-1:
                layers += [nn.Linear(hidden, 3072), nn.BatchNorm1d(num_features=3072), nn.ReLU()]
        
        self.layers = nn.Sequential(*layers)
        print(fc_bias)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(3, 32, 32)
        return x
    