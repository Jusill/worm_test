import torch
import torch.nn as nn
from collections import OrderedDict
from Utils import WORM_RECURRENT_VIEW


class WormNET(nn.Module):
    def __init__(self):
        super(WormNET, self).__init__()

        # in - (batch_size, 12 - depth, 11 - width, 12 - height)
        self.features = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3*(WORM_RECURRENT_VIEW + 1), 16, 3)),
        ('prelu1', nn.PReLU(16)),
        ('conv2', nn.Conv2d(16, 32, 3)),
        ('prelu2', nn.PReLU(32)),
        ('conv3', nn.Conv2d(32, 48, 3)),
        ('prelu3', nn.PReLU(48)),
        ('conv4', nn.Conv2d(48, 64, 3)),
        ('prelu4', nn.PReLU(64)),
        ('conv5', nn.Conv2d(64, 80, 3)),
        ('prelu5', nn.PReLU(80))]))
        # out - (batch_size, 160, 1, 2)

        # in - (batch_size, 1, 160, 2)
        self.action = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(2, 24)),
        ('prelu1', nn.PReLU(1)),
        ('pool1', nn.AvgPool2d(2)),
        ('linear2', nn.Linear(12, 30)),
        ('prelu2', nn.PReLU(1)),
        ('pool2', nn.AvgPool2d(2)),
        ('linear3', nn.Linear(15, 34)),
        ('prelu3', nn.PReLU(1)),
        ('pool3', nn.AvgPool2d(2)),
        ('linear4', nn.Linear(17, 36)),
        ('prelu4', nn.PReLU(1)),
        ('pool4', nn.AvgPool2d(2)),
        ('pool5', nn.MaxPool2d((10, 1))),
        ('softmax', nn.Softmax(dim=3))
        ]))
        # out - (batch_size, 1, 1, 18)

    def forward(self, x):
        features = self.features(x)
        features = features.permute(0, 2, 1, 3)
        prob = self.action(features)

        return prob
