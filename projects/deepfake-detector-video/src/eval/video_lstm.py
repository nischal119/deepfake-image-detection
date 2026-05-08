
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetLSTMVideoClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = resnet18(weights=None)
            
        self.in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone
        
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        batch_size, time_steps, channels, height, width = x.size()
        
        x = x.view(batch_size * time_steps, channels, height, width)
        
        feats = self.feature_extractor(x)
        
        feats = feats.view(batch_size, time_steps, -1)
        
        lstm_out, _ = self.lstm(feats)
        
        last_out = lstm_out[:, -1, :]   
        
          
        logits = self.classifier(last_out)
        
        return logits
