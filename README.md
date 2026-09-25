# OPG Multi-Label Classification

Multi-label classification of dental findings in panoramic radiographs (OPGs) using a pretrained ResNet18 model.

This project develops an image-level deep learning pipeline for predicting multiple dental findings simultaneously from a complete panoramic dental radiograph.

> **Note:** This project is intended for research and educational purposes and is not a validated clinical diagnostic system.

## Project Overview

Panoramic dental radiographs may contain several dental findings at the same time. Therefore, the task is formulated as a **multi-label classification problem**, where multiple findings can be present in a single OPG.

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

**DOI:** `10.5281/zenodo.15487430`

The main dataset contains **1,628 OPGs** divided into:

- Training set: **1,375 OPGs**
- Validation set: **153 OPGs**
- Internal test set: **100 OPGs**

An additional external validation set containing **180 OPGs** is also evaluated.

The original annotations are provided in YOLO format:

```text
class_id x_center y_center width height
```

Although the dataset contains bounding-box annotations, this project reformulates the problem as **image-level multi-label classification**.

For each OPG, the unique class IDs are extracted from its annotation file and converted into a 14-dimensional multi-hot target vector.

Bounding-box coordinates are not used during classifier training, but they are retained for visualization and post-hoc error analysis.

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

## Exploratory Data Analysis

The dataset was analyzed to examine:

- Class frequency
- Class prevalence
- Number of findings per OPG
- Distribution differences across training, validation, and test splits
- Class imbalance

The analysis showed substantial imbalance between common and rare dental findings.

Some classes contain only a small number of positive examples, which limits the reliability of their evaluation metrics.

## Image Preprocessing

Each OPG is:

1. Resized to `224 × 224`
2. Converted to a PyTorch tensor
3. Normalized using ImageNet mean and standard deviation

Although panoramic radiographs are inherently grayscale, they are represented as three-channel RGB images to maintain compatibility with the ImageNet-pretrained ResNet18 architecture.

No data augmentation is used in the current version of the project.

## Model Architecture

A pretrained **ResNet18** is used as the backbone.

The original ImageNet classification head is replaced with a fully connected layer containing **14 outputs**, one for each dental finding.

Because this is a multi-label problem, the outputs are treated independently using sigmoid probabilities rather than a softmax function.

## Training Strategy

Training is performed in three stages.

### 1. Baseline Transfer Learning

The pretrained ResNet18 backbone is frozen and only the final classification layer is trained.

### 2. Fine-Tuning

The final ResNet block (`layer4`) and classification head are jointly optimized using a smaller learning rate.

The checkpoint with the lowest validation loss is retained.

### 3. Class-Weighted Fine-Tuning

The dataset contains substantial class imbalance, so positive-class weights are incorporated into `BCEWithLogitsLoss`.

To avoid excessively large weights for very rare classes:

- weights below 1 are replaced by 1
- a square-root transformation is applied
- the final weights are capped at 10

The weighted model is initialized from the best fine-tuned checkpoint.

Model selection during weighted fine-tuning is based on **validation Macro AUPRC**.

## Threshold Optimization

A fixed threshold of `0.5` may not be optimal for every dental finding.

Therefore, a separate decision threshold is optimized for each class using the **validation set only**.

Candidate thresholds from `0.05` to `0.95` are evaluated, and the threshold producing the highest class-specific validation F1 score is selected.

The internal test and external validation sets are not used for threshold optimization.

For classes without positive validation samples, a default threshold of `0.5` is retained.

## Internal Test Evaluation

The final weighted model is evaluated on the predefined internal test set using the validation-derived class-specific thresholds.

### Overall Internal Test Performance

| Metric | Score |
|---|---:|
| Micro F1 | 0.675 |
| Macro F1 | 0.515 |
| Macro F1 (supported classes) | 0.555 |

Performance varies substantially across dental findings, and results for very low-support classes should be interpreted cautiously.

## External Validation

The final model is also evaluated on the additional external validation dataset.

The external dataset is not used for:

- model training
- model selection
- threshold optimization

The model weights and class-specific thresholds remain fixed during external evaluation.

### Overall External Performance

| Metric | Score |
|---|---:|
| Micro F1 | 0.625 |
| Macro F1 | 0.371 |
| Macro F1 (supported classes) | 0.400 |

The reduction in external Macro F1 is larger than the reduction in Micro F1, indicating that external performance degradation is not uniform across classes.

Several findings maintain relatively strong performance, while others show substantially weaker external generalization.

## Error Analysis

A focused error analysis is performed for **Apical periodontitis** on the external validation set.

The analysis investigates:

- True positives
- False positives
- False negatives
- Bounding-box size
- Number of annotated findings per OPG
- Spatial distribution of annotations

The analysis showed that:

- Both false positives and false negatives occur frequently.
- False negative cases do not show a clear difference in bounding-box size compared with true positive cases.
- True positive OPGs contain more annotated Apical periodontitis boxes on average, although the median number of boxes is the same.
- The spatial distributions of true positive and false negative annotations are broadly similar.

These observations suggest that no single simple property such as lesion size or image location fully explains the observed classification errors.

## Inference

The final model checkpoint contains:

- Model weights
- Class-specific thresholds
- Class names
- Input image size

During inference:

1. The OPG is loaded and preprocessed.
2. The model produces 14 logits.
3. Sigmoid converts the logits to probabilities.
4. Validation-derived thresholds convert probabilities into binary predictions.

The predictions indicate whether each finding is present at the **image level** and do not provide spatial localization.

## Repository Contents

```text
opg-multilabel-classification/
│
├── opg_multilabel_classification.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

The following local directories are intentionally excluded from version control:

```text
data/
models/
.ipynb_checkpoints/
```

## Local Project Structure

A local setup similar to the following can be used:

```text
opg-multilabel-classification/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── test_alte_cabinete/
│       └── Ext-validation/
│           ├── images/
│           └── labels/
│
├── models/
├── results/
└── notebooks/
```

## Requirements

The project uses:

- Python
- PyTorch
- torchvision
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook

## Running the Project

1. Clone or download this repository.
2. Obtain the dataset separately from the original Zenodo source.
3. Place the dataset in the expected local `data/` structure.
4. Open `opg_multilabel_classification.ipynb` using Jupyter Notebook.
5. Execute the notebook cells sequentially.

The dataset and trained model checkpoints are intentionally not distributed through this repository.

## Limitations

This project has several important limitations:

- The task is image-level classification and does not localize individual dental findings.
- Panoramic radiographs are directly resized to `224 × 224`, changing their original aspect ratio and potentially introducing geometric distortion.
- The dataset is highly imbalanced.
- Some findings contain very few positive samples.
- Root resorption cannot be meaningfully evaluated because no positive samples are available in the validation, internal test, or external validation splits.
- Metrics for very low-support classes are unstable and should be interpreted cautiously.
- The class-weighting strategy uses a heuristic square-root transformation and upper cap.
- Decision thresholds are optimized for validation F1 and may not represent the optimal operating point for clinical use.
- External performance varies substantially across dental findings.
- Additional evaluation on independent datasets is required before considering any clinical application.

## Future Work

Possible extensions include:

- Aspect-ratio-preserving image preprocessing
- Stronger data augmentation
- Alternative pretrained architectures
- Probability calibration
- Object detection
- Tooth-level classification
- Segmentation
- Evaluation on additional independent datasets
- Statistical confidence intervals and uncertainty analysis

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

The dataset used in this project is **not included in this repository** and remains subject to the terms of use specified by the original dataset authors on Zenodo.

Please refer to the original dataset source for its licensing and usage conditions.
