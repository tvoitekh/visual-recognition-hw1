"""
Model architectures for plant classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34, resnet50, resnet18,
    ResNet34_Weights, ResNet50_Weights, ResNet18_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights
)


class PlantClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=100,
                 pretrained=True, dropout_rate=0.3):
        super(PlantClassifier, self).__init__()

        # Initialize backbone based on name
        if backbone_name == 'resnet18':
            if pretrained:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.backbone = backbone
            else:
                self.backbone = resnet18(weights=None)
        elif backbone_name == 'resnet34':
            if pretrained:
                backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                self.backbone = backbone
            else:
                self.backbone = resnet34(weights=None)
        elif backbone_name == 'resnet50':
            if pretrained:
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.backbone = backbone
            else:
                self.backbone = resnet50(weights=None)
        elif backbone_name == 'resnext50':
            if pretrained:
                backbone = resnext50_32x4d(
                    weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
                self.backbone = backbone
            else:
                self.backbone = resnext50_32x4d(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Get final layer's input features
        in_features = self.backbone.fc.in_features

        # Replace the final fully connected layer with a custom head
        self.backbone.fc = nn.Identity()

        # Create a new classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Initialize the new layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        # Apply classifier head
        return self.classifier(features)


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p),
                            (x.size(-2), x.size(-1))).pow(1./p)


class AdvancedPlantClassifier(nn.Module):
    def __init__(self, backbone_name,
                 num_classes=100, pretrained=True, dropout_rate=0.3):
        super(AdvancedPlantClassifier, self).__init__()

        # Initialize backbone based on name (without the final FC layer)
        if backbone_name == 'resnet18':
            if pretrained:
                base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet18(weights=None)
        elif backbone_name == 'resnet34':
            if pretrained:
                base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet34(weights=None)
        elif backbone_name == 'resnet50':
            if pretrained:
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                base_model = resnet50(weights=None)
        elif backbone_name == 'resnext50':
            if pretrained:
                base_model = resnext50_32x4d(
                    weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            else:
                base_model = resnext50_32x4d(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Extract all layers except the final FC
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Get the number of features from the backbone
        if backbone_name in ['resnet18', 'resnet34']:
            in_features = 512
        else:  # resnet50, resnext50
            in_features = 2048

        # Global Generalized Mean Pooling
        self.global_pool = GeM()

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Initialize the new layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.channel_attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        features = self.features(x)

        # Apply pooling
        pooled = self.global_pool(features).view(x.size(0), -1)

        # Apply channel attention
        att = self.channel_attention(pooled)
        enhanced = pooled * att

        # Apply classifier
        return self.classifier(enhanced)
