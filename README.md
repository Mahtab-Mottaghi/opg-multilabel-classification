# OPG Multi-Label Classification

Multi-label classification of dental findings in panoramic radiographs (OPGs) using a pretrained ResNet18 model.

This project develops an image-level deep learning pipeline for predicting multiple dental findings simultaneously from a complete panoramic dental radiograph.

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

The dataset used in this project is **not included in this repository**. The dataset remains subject to the terms of use specified by the original dataset authors on Zenodo.

Please refer to the original dataset source for its licensing and usage conditions.

## Project Overview

Panoramic dental radiographs may contain several findings at the same time.  
Therefore, the task is formulated as a **multi-label classification problem**, where each OPG can contain multiple positive labels.

The project includes:

- Conversion of YOLO annotations into image-level multi-label targets
- Exploratory data analysis
- Image preprocessing
- Transfer learning with ResNet18
- Fine-tuning of the final ResNet block
- Class imbalance handling using weighted binary cross-entropy
- Per-class threshold optimization
- Internal test evaluation
- External validation
- Error analysis
- Single-image inference

## Dataset

This project uses the public dataset:

**Dataset for Automating Dental Condition Detection on Panoramic Radiographs**

DOI: `10.5281/zenodo.15487430`

The main dataset contains:

- 1,375 training OPGs
- 153 validation OPGs
- 100 internal test OPGs

An additional external validation set containing 180 OPGs is also evaluated.

The original annotations are provided in YOLO format:

`class_id x_center y_center width height`

Although the dataset contains bounding-box annotations, this project reformulates the task as **image-level multi-label classification**.

Only the unique class IDs from each OPG are used during classifier training. Bounding-box coordinates are retained only for visualization and error analysis.

The dataset itself is **not included in this repository**.

## Target Findings

The model predicts the following 14 dental findings:

1. Implant
2. Prosthetic restoration
3. Obturation
4. Endodontic treatment
5. Carious lesion
6. Bone resorption
7. Impacted tooth
8. Apical periodontitis
9. Root fragment
10. Furcation lesion
11. Apical surgery
12. Root resorption
13. Orthodontic device
14. Surgical device

## Method

A pretrained **ResNet18** is used as the backbone.

The original ImageNet classification head is replaced with a 14-output fully connected layer.

Training is performed in three stages:

1. **Baseline transfer learning**  
   Only the final classification layer is trained.

2. **Fine-tuning**  
   The final ResNet block (`layer4`) and classification head are trained together.

3. **Class-weighted fine-tuning**  
   Positive-class weights are introduced to reduce the effect of class imbalance.

The model uses:

- `BCEWithLogitsLoss`
- Adam optimizer
- Image size: `224 × 224`
- ImageNet normalization
- Per-class decision thresholds optimized on the validation set

## Threshold Optimization

A fixed threshold of 0.5 may not be optimal for every dental finding.

Therefore, a separate threshold is selected for each class using the validation set by maximizing the class-specific F1 score.

The internal test and external validation sets are not used for threshold optimization.

## External Validation

The final model is evaluated on the additional external validation dataset without modifying:

- model weights
- training configuration
- decision thresholds

This provides an additional assessment of model generalization beyond the main train/validation/test splits.

## Error Analysis

A focused error analysis is performed for **Apical periodontitis**.

The analysis investigates:

- False positive cases
- False negative cases
- Bounding-box size
- Number of annotated findings per OPG
- Spatial distribution of annotations

The results suggest that no single simple property such as lesion size or image location fully explains the observed classification errors.

## Repository Contents

```text
opg-multilabel-classification/
│
├── opg_multilabel_classification.ipynb
├── README.md
├── LICENSE
└── .gitignore


