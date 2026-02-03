import torch
import torch.nn as nn
from src.models.backbone_stub import ResNetBackboneStub
from src.blocks.residual_sge import BottleneckSGE

class SGENetStub(nn.Module):
    def __init__(self, layers=(3,4,6,3), num_classes=1000, groups_sge=64, use_sge_stages=(1,2,3,4)):
        super().__init__()
        self.backbone = ResNetBackboneStub(layers=layers, groups_sge=groups_sge)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * BottleneckSGE.expansion, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        f4 = feats['stage4']
        out = self.avgpool(f4)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        feats['logits'] = logits
        return feats
