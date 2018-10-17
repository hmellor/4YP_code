import functools

import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d

# FCN-AlexNet
class fcnan(nn.Module):
    def __init__(self, n_classes=21):
        super(fcnan, self).__init__()
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d,
                                      size_average=False)

        self.features = nn.Sequential(
            # Base net
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Caffe has LRN here
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Caffe has LRN here
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        self.classifier = nn.Sequential(
            # Fully Conv
            nn.Conv2d(256, 4096, kernel_size=6, padding=0),
            nn.ReLU(inplace=False),
            nn.Dropout(inplace=False),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.ReLU(inplace=False),
            nn.Dropout(inplace=False),
            nn.Conv2d(4096, self.n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        basenet = self.features(x)
        # x = x.view(x.size(0), 256 * 16 * 20)
        score = self.classifier(basenet)
        out = F.interpolate(score, x.size()[2:])
        return out



    def init_alexnet_params(self, alexnet, copy_fc8=True):
        features = list(alexnet.features.children())

        for idx, feature in enumerate(features):
            l1 = features[idx]
            if isinstance(l1, nn.Conv2d) and isinstance(feature, nn.Conv2d):
                assert l1.weight.size() == feature.weight.size()
                assert l1.bias.size() == feature.bias.size()
                feature.weight.data = l1.weight.data
                feature.bias.data = l1.bias.data


        # ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        # features = list(alexnet.features.children())
        #
        # for idx, conv_block in enumerate(blocks):
        #     for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
        #         if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        #             assert l1.weight.size() == l2.weight.size()
        #             assert l1.bias.size() == l2.bias.size()
        #             l2.weight.data = l1.weight.data
        #             l2.bias.data = l1.bias.data
        # for i1, i2 in zip([0, 3], [0, 3]):
        #     l1 = alexnet.classifier[i1]
        #     l2 = self.classifier[i2]
        #     l2.weight.data = l1.weight.data.view(l2.weight.size())
        #     l2.bias.data = l1.bias.data.view(l2.bias.size())
        # n_class = self.classifier[6].weight.size()[0]
        # if copy_fc8:
        #     l1 = alexnet.classifier[6]
        #     l2 = self.classifier[6]
        #     l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
        #     l2.bias.data = l1.bias.data[:n_class]
