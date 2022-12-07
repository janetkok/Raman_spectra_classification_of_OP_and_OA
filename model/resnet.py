""" model 
ResNet Model adapted from https://github.com/csho33/bacteria-ID/blob/master/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import copy
import numpy as np
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,input_dim=1000, in_channels=64, n_classes=30):
        super(ResNet, self).__init__()

        layers = 6
        hidden_size = 100
        block_size = 2
        hidden_sizes = [hidden_size] * layers
        num_blocks = [block_size] * layers

        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()
        self.linear = nn.Linear(self.z_dim, self.n_classes)


    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.linear(z)


    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim

class Multi_ResNet(nn.Module):
    def __init__(self, input_dim=1000,
        in_channels=64, n_classes=2,channel=1):
        super(Multi_ResNet, self).__init__()
        layers = 6
        hidden_size = 100
        block_size = 2
        hidden_sizes = [hidden_size] * layers
        num_blocks = [block_size] * layers
        assert len(num_blocks) == len(hidden_sizes)
        
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # load pretrain
        pretrain_input_dim = 1000
        pretrain_n_classes = 30
        self.pretrain_model = ResNet(input_dim=pretrain_input_dim, n_classes=pretrain_n_classes)
        self.pretrain_model.load_state_dict(torch.load(
            './model/pretrained_model.ckpt', map_location=lambda storage, loc: storage))
        tempMod = ResNet(input_dim=input_dim,n_classes=n_classes)
        time.sleep(3) #waiting to load pretrain model 

        # replace final classification layer
        self.pretrain_model.linear=nn.Linear(tempMod.linear.in_features, n_classes)

        # modify initial convolutional layer to take in 8-channel data (by duplicating the weights)
        conv1_w = self.pretrain_model.conv1.weight
        conv1_w = conv1_w.repeat_interleave(channel, dim=1) 
        self.pretrain_model.conv1 =  nn.Conv1d(channel, self.in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.pretrain_model.conv1.weight = nn.Parameter(conv1_w)

    def forward(self, x):
        z = self.pretrain_model(x)
        return z
