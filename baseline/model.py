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

# Resnet

class resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

class resnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class resnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class resnext50_32x4d(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class resnext101_32x8d(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class wide_resnet50_2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

class wide_resnet101_2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.wide_resnet101_2(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)
    def forward(self, x):
        return self.model(x)

#VGG

class vgg11(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg11(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg11_bn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg11_bn(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg13(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg13(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg13_bn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg13_bn(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg16(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg16_bn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg16_bn(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg19(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

class vgg19_bn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.vgg19_bn(pretrained=True)
        self.model.classifier= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)

#AlexNet

class alexnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model=torchvision.models.alexnet(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.model(x)

#ViT

class vit(nn.Module):
    def __init__(self, num_classes,image_size):
        super().__init__()
        self.model=vit_pytorch.ViT( 
            image_size = image_size,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1)

    def forward(self, x):
        return self.model(x)

from vit_pytorch.deepvit import DeepViT

class deepvit(nn.Module):
    def __init__(self, num_classes,image_size):
        super().__init__()
        self.model=DeepViT(
        image_size = image_size,
        patch_size = 32,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
)

    def forward(self, x):
        return self.model(x)

