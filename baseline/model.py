import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


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

