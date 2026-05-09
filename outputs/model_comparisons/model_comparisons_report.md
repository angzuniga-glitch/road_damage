# Road Damage Detection — Model Comparison Report

## Baseline approach

Scratch variants (no pretrained weights) serve as the baseline for each model
family. This isolates the contribution of ImageNet pretraining and end-to-end
fine-tuning directly from the RDD2022 dataset, without reference to external
literature benchmarks.

## Class definitions

All macro metrics are averaged equally across these 4 classes:

| Class | Description |
|-------|-------------|
```
{'D00': 'Longitudinal cracking  — cracks parallel to road direction', 'D10': 'Transverse cracking    — cracks perpendicular to road direction', 'D20': 'Alligator cracking     — interconnected mesh-pattern cracks', 'D40': 'Pothole                — bowl-shaped holes in road surface'}
```

## All models comparison

```
----------------------------------------------------------------------------------------------------
Model                           Precision     Recall         F1     mAP50*  Notes
----------------------------------------------------------------------------------------------------

  FASTER R-CNN
  [BASE] Scratch                   0.3738     0.3360     0.3424     0.2049  baseline (no pretraining)
         Frozen                    0.2461     0.2948     0.2665     0.1404  
         Finetune                  0.4078     0.6358     0.4940     0.4749  best variant


  RESNET-18
  [BASE] Scratch                   0.9094     0.9193     0.9094     0.9094  baseline (no pretraining)
         Frozen                    0.7942     0.8191     0.7942     0.7942  
         Finetune                  0.9347     0.9399     0.9347     0.9347  best variant


  VIT
  [BASE] Scratch                   0.8171     0.8444     0.8171     0.8171  baseline (no pretraining)
         Frozen                    0.7542     0.7787     0.7542     0.7542  
         Finetune                  0.9762     0.9772     0.9762     0.9762  best variant


  YOLOV8N
  [BASE] Scratch                   0.8604     0.2556     0.0000     0.2355  baseline (no pretraining)
         Frozen                    0.8458     0.1978     0.0000     0.1804  
         Finetune                  0.8425     0.2940     0.0000     0.2721  best variant


  CUSTOM CNN
         CustomCNN                 0.8124     0.8436     0.8124     0.8124  baseline (no pretraining)

----------------------------------------------------------------------------------------------------
```

## Ablation (scratch to frozen to finetune)

```

  FASTER R-CNN — Training mode ablation  (metric: map50)
  Mode                    Value  delta vs scratch
  ---------------------------------------------
  Scratch (baseline)     0.2049               —
  Frozen backbone        0.1404         -0.0644
  Full finetune          0.4749         +0.2700

  RESNET-18 — Training mode ablation  (metric: map50)
  Mode                    Value  delta vs scratch
  ---------------------------------------------
  Scratch (baseline)     0.9094               —
  Frozen backbone        0.7942         -0.1152
  Full finetune          0.9347         +0.0253

  VIT — Training mode ablation  (metric: map50)
  Mode                    Value  delta vs scratch
  ---------------------------------------------
  Scratch (baseline)     0.8171               —
  Frozen backbone        0.7542         -0.0629
  Full finetune          0.9762         +0.1591

  YOLOV8N — Training mode ablation  (metric: map50)
  Mode                    Value  delta vs scratch
  ---------------------------------------------
  Scratch (baseline)     0.2355               —
  Frozen backbone        0.1804         -0.0551
  Full finetune          0.2721         +0.0366
```

## Per-class breakdown across finetuned models

```
----------------------------------------------------------------------------------------------------
  Per-class AP50 / F1 — finetune models only
  D00: Longitudinal cracking  — cracks parallel to road direction
  D10: Transverse cracking    — cracks perpendicular to road direction
  D20: Alligator cracking     — interconnected mesh-pattern cracks
  D40: Pothole                — bowl-shaped holes in road surface
----------------------------------------------------------------------------------------------------
  Model                             D00        D10        D20        D40
----------------------------------------------------------------------------------------------------
  fasterrcnn_finetune            0.5024     0.4933     0.5516     0.3521
  yolo_finetune                  0.2860     0.2776     0.3605     0.1642
----------------------------------------------------------------------------------------------------
```
