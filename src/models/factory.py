from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
)

import timm

from src.models.custom_cnn import CustomCNN


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_resnet(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
    elif name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = tvm.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet variant: {name}")

    in_features = model.fc.in_features

    if freeze_backbone:
        # Freeze everything first
        freeze_module(model)
        # Replace classifier with trainable head
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def _build_vit(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Uses timm ViT models.
    Good starter names:
      - vit_tiny_patch16_224
      - vit_small_patch16_224
      - vit_base_patch16_224
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        freeze_module(model)

        # Re-enable classification head only
        # timm ViTs usually expose .head
        if hasattr(model, "head") and isinstance(model.head, nn.Module):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {name} does not expose a standard .head for freezing/fine-tuning.")

    return model


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory for all required model families.

    Supported:
      - custom_cnn
      - resnet18
      - resnet34
      - vit_tiny_patch16_224
      - vit_small_patch16_224
      - vit_base_patch16_224

    Modes:
      - scratch: pretrained=False, freeze_backbone=False
      - frozen pretrained: pretrained=True, freeze_backbone=True
      - fine-tuned pretrained: pretrained=True, freeze_backbone=False
    """
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        if freeze_backbone:
            raise ValueError("freeze_backbone is not supported for custom_cnn.")
        return CustomCNN(num_classes=num_classes)

    if model_name in {"resnet18", "resnet34"}:
        return _build_resnet(
            name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if model_name in {
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
    }:
        return _build_vit(
            name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


if __name__ == "__main__":
    configs = [
        ("custom_cnn", False, False),
        ("resnet18", False, False),
        ("resnet18", True, True),
        ("resnet18", True, False),
        ("vit_tiny_patch16_224", False, False),
        ("vit_tiny_patch16_224", True, True),
        ("vit_tiny_patch16_224", True, False),
    ]

    for model_name, pretrained, freeze_backbone in configs:
        print("=" * 80)
        print(
            f"model={model_name}, pretrained={pretrained}, freeze_backbone={freeze_backbone}"
        )
        model = create_model(
            model_name=model_name,
            num_classes=8,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        print(model.__class__.__name__)
        print(f"trainable params: {count_trainable_parameters(model):,}")