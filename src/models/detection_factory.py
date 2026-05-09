from __future__ import annotations

import logging

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

logger = logging.getLogger(__name__)

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_detection_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """
    num_classes includes background for torchvision detectors.
    If you have 4 classes, pass num_classes=5.
    """
    model_name = model_name.lower()

    if model_name != "fasterrcnn_resnet50_fpn":
        raise ValueError(f"Unsupported detection model: {model_name}")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    # 0 trainable layers = frozen backbone
    # 5 trainable layers = full backbone fine-tuning
    trainable_backbone_layers = 0 if freeze_backbone else 5

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        # torchvision already respects trainable_backbone_layers,
        # but this makes intent explicit.
        for param in model.backbone.body.parameters():
            param.requires_grad = False

        logger.info(
        "Created %s | pretrained=%s | freeze_backbone=%s | num_classes=%s | trainable_params=%s",
        model_name,
        pretrained,
        freeze_backbone,
        num_classes,
        f"{count_trainable_parameters(model):,}",
    )  # pylint: disable=logging-too-many-args

    return model
