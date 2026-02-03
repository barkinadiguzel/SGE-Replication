import torch
import torch.nn as nn
from src.blocks.residual_sge import BottleneckSGE

def _make_downsample(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )

class ResNetBackboneStub(nn.Module):
    def __init__(self, layers=(3,4,6,3), groups_sge=64, eps=1e-5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1, groups_sge=groups_sge, eps=eps)
        self.layer2 = self._make_layer(128, layers[1], stride=2, groups_sge=groups_sge, eps=eps)
        self.layer3 = self._make_layer(256, layers[2], stride=2, groups_sge=groups_sge, eps=eps)
        self.layer4 = self._make_layer(512, layers[3], stride=2, groups_sge=groups_sge, eps=eps)

    def _make_layer(self, planes, blocks, stride=1, groups_sge=64, eps=1e-5):
        downsample = None
        outplanes = planes * BottleneckSGE.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = _make_downsample(self.inplanes, outplanes, stride)

        layers = []
        layers.append(BottleneckSGE(self.inplanes, planes, stride, groups_sge, downsample, eps=eps))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(BottleneckSGE(self.inplanes, planes, groups_sge=groups_sge, downsample=None, eps=eps))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return {
            'stage1': f1,
            'stage2': f2,
            'stage3': f3,
            'stage4': f4,
        }
