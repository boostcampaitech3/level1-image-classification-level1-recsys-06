from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet

# Custom Model Template

class densenet121(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1024, out_features=1000, bias=True)
        self.model.classifier = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.model(x)

class densenet161(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.model = torchvision.models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(in_features=2208, out_features=1000, bias=True)
        self.model.classifier = nn.Linear(2208, num_classes)
    def forward(self, x):
        return self.model(x)

class densenet169(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.model = torchvision.models.densenet169(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1664, out_features=1000, bias=True)
        self.model.classifier = nn.Linear(1664, num_classes)
    def forward(self, x):
        return self.model(x)

class densenet201(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.model = torchvision.models.densenet201(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1920, out_features=1000, bias=True)
        self.model.classifier = nn.Linear(1920, num_classes)
    def forward(self, x):
        return self.model(x)


## Efficientnet
class efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b6(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)

class efficientnet_b7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)

    def forward(self, x):
        
        return self.model(x)



class googlenet(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.googlenet(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.dropout = nn.Dropout(0.2)
        self.model.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.model(x)

