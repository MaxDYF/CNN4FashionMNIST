from tkinter.dialog import DIALOG_ICON

import torch
from const_args import *

class CNNet(torch.nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv_layer1 = torch.nn.Sequential(
            # Input 1*28*28
            # Output
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=CONV1_OUTPUT_CHANNELS,
                kernel_size=KERNEL_SIZE.tolist(),
                stride=KERNEL_STRIDE,
                dilation=DILATION,
                padding=PADDING_SIZE.tolist()
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=POOL_SIZE.tolist(), stride=POOL_STRIDE),
        )
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=CONV1_OUTPUT_CHANNELS,
                out_channels=CONV2_OUTPUT_CHANNELS,
                kernel_size=KERNEL_SIZE.tolist(),
                stride=KERNEL_STRIDE,
                dilation=DILATION,
                padding=PADDING_SIZE.tolist()
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=POOL_SIZE.tolist(), stride=POOL_STRIDE),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(
                FC_INPUT_SIZE,
                64
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.reshape(-1, FC_INPUT_SIZE)
        x = self.fc_layer(x)
        return x