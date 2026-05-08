 

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18VideoClassifier(nn.Module):
  

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        input_clip_length: int = 16,
    ) -> None:
        super().__init__()
        self.input_clip_length = input_clip_length

        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            backbone = r3d_18(weights=weights)
        else:
            backbone = r3d_18(weights=None)

          
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)

        Returns:
            logits: (B, num_classes)
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
