import torch
import torch.nn as nn
from torchvision import models
from sklearn.linear_model import LinearRegression

class ResNet50Regressor(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze early layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze last layers
        for param in list(self.resnet.layer3.parameters()) + list(self.resnet.layer4.parameters()):
            param.requires_grad = True

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.float()
        return self.resnet(x)

def train_linear_model(train_df, intercept=True):
    """Train a linear regression model given white pixel percentages and JSW."""
    X_train = train_df[["White_Pixel_Percentage"]]
    y_train = train_df["JSW"]
    model = LinearRegression(fit_intercept=intercept)
    model.fit(X_train, y_train)
    return model