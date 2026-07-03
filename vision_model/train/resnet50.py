import timm
import torch
import torch.nn as nn
from typing import Any
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        model = models.resnet50("IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, 10)
        self.net = model

    def forward(self, x):
        return self.net(x)
    
