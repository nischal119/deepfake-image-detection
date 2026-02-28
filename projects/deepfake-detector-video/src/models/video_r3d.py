"""R3D-18 video classifier for binary deepfake detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18VideoClassifier(nn.Module):
    """
    3D ResNet-18 for video deepfake classification.

    Uses torchvision.models.video.r3d_18 with ImageNet Kinetics pretrained weights.
    Replaces the final fc layer for binary (real/fake) classification.

    Input: (B, C, T, H, W) — batch, channels, time (frames), height, width.
    Typical: (B, 3, 16, 112, 112) or (B, 3, 16, 224, 224).

    Output: (B, num_classes) logits.
    """

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

        # r3d_18 has .fc as final layer
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
