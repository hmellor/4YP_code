import math
import functools

import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d


class fcalexnet(nn.Module):
    """
    Implementation adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """

    def __init__(self, n_classes=21):
        super(fcalexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1),
        )
        self.deconv = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=63,
                                         stride=32, bias=False)

        self.min_size = 224

    def pad_input(self, x):
        """
        Pad input with zeros if one of the input spatial dimensions is lower than min_size
        """
        padding = [0, 0, 0, 0]
        if x.size(2) < self.min_size:
            # padTop, padBottom
            padding[2] = padding[3] = math.ceil((self.min_size - x.size(2)) / 2.)
        if x.size(3) < self.min_size:
            # padLeft, padRight
            padding[0] = padding[1] = math.ceil((self.min_size - x.size(3)) / 2.)
        if sum(padding):
            x = F.pad(x, padding)
        return x

    def forward(self, x):
        x_padded = self.pad_input(x)
        out_features = self.features(x_padded)
        out_classifier = self.classifier(out_features)
        out_deconv = self.deconv(out_classifier)
        out_interpolated = F.interpolate(out_deconv, x.size()[2:])
        return out_interpolated

    def init_alexnet_params(self, alexnet):

        features_rand = [m for m in self.features if not isinstance(m, nn.LocalResponseNorm)]
        features_trained = [m for m in alexnet.features if not isinstance(m, nn.LocalResponseNorm)]
        for (new_module, trained_module) in zip(features_rand, features_trained):
            if not isinstance(new_module, nn.Conv2d):
                continue
            assert isinstance(trained_module, nn.Conv2d)
            new_module.load_state_dict(trained_module.state_dict())

        classifier_rand = [m for m in self.classifier if not isinstance(m, nn.Dropout)]
        classifier_trained = [m for m in alexnet.classifier if not isinstance(m, nn.Dropout)]
        for (new_module, trained_module) in list(zip(classifier_rand, classifier_trained))[:-1]:
            if not isinstance(new_module, nn.Conv2d):
                continue
            assert isinstance(trained_module, nn.Linear)
            trained_state = trained_module.state_dict()
            trained_state['weight'] = trained_state['weight'].view(new_module.weight.size())
            new_module.load_state_dict(trained_state)
