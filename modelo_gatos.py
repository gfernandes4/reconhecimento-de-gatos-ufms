import torch
import torch.nn as nn
import torchvision.models as models

class GatosCNN(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super(GatosCNN, self).__init__()
        
        # Carrega ResNet18 pré-treinado no ImageNet
        self.model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Substitui a última camada para ajustar ao número de classes
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model_ft(x)
