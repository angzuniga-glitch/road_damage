from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Clean CNN for crop-level road-damage classification.

    Input:
        (B, 3, H, W), typically (B, 3, 224, 224)

    Architecture:
        4 convolutional stages
        Global average pooling
        Small MLP classifier head
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        dropout_features: float = 0.10,
        dropout_classifier: float = 0.30,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout=0.0),               # 224 -> 112
            ConvBlock(32, 64, dropout=dropout_features),           # 112 -> 56
            ConvBlock(64, 128, dropout=dropout_features),          # 56 -> 28
            ConvBlock(128, 256, dropout=dropout_features),         # 28 -> 14
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 256, 1, 1) -> (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_classifier),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CustomCNN(num_classes=8)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)

    print(model)
    print(f"Output shape: {y.shape}")   # expected: (4, 8)
    print(f"Trainable params: {count_parameters(model):,}")