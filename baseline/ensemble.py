#Ensemble
import torch.nn as nn
import torch.nn.functional as F
import torch


class myensemble2(nn.Module):
    def __init__(self, modelA, modelB,num_classes):
        super(myensemble2, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(num_classes*2, num_classes)

    def forward(self,x):
        x1=self.modelA(x)
        x2=self.modelB(x)
        x=torch.cat((x1,x2),dim=1)
        x=self.classifier(F.relu(x))
        return x

class myensemble3(nn.Module):
    def __init__(self, modelA, modelB,modelC,num_classes):
        super(myensemble3, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(num_classes*3, num_classes)

    def forward(self,x):
        x1=self.modelA(x)
        x2=self.modelB(x)
        x3=self.modelC(x)
        x=torch.cat((x1,x2,x3),dim=1)
        x=self.classifier(F.relu(x))
        return x

class myensemble4(nn.Module):
    def __init__(self, modelA, modelB,modelC,modelD,num_classes):
        super(myensemble4, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.classifier = nn.Linear(num_classes*4, num_classes)

    def forward(self,x):
        x1=self.modelA(x)
        x2=self.modelB(x)
        x3=self.modelC(x)
        x4=self.modelD(x)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        x=self.classifier(F.relu(x))
        return x


