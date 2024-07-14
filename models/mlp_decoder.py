import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_DECODER(nn.Module):
    def __init__(self, hidden, depth = 6, fc_bias = True, num_classes = 10):
        # Depth means how many layers before final linear layer
        
        super(MLP_DECODER, self).__init__()
        
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        layers = [nn.Linear(hidden, 3072), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_classes, hidden, bias = fc_bias)
        print(fc_bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.layers(x)
        return x
    