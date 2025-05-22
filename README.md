Using the ACDC dataset (100 subjects with ground truth annotations), you'll implement and compare deep learning models for medical image segmentation. The provided codebase includes starter templates and supports training on GPUs/Colab.

## Key Tasks

### 1. Model Architectures
- **YourUNet**: Modify the provided UNet with ≥3 new features (e.g., residual layers, multi-resolution outputs)
- **YourSegNet**: Design an original asymmetric architecture (non-encoder/decoder)
- Implement custom loss functions (beyond cross-entropy)
- *Validation accuracy target*: ≥75%

### 2. Checkpointing
Implement real-time saving of the best validation model during training.

### 3. Data Augmentation
Add two augmentation techniques (e.g., rotations, elastic deformations) to improve generalization.

### 4. Report
Structured technical report covering:
- Architecture diagrams and modifications
- Training curves analysis
- Augmentation examples
- Exact command-line instructions

## Technical Requirements
- **Data Format**: HDF5 files (pre-processed ACDC dataset)
- **Framework**: PyTorch
- **GPU Support**: Required (Colab or local GPU)
- **Code Quality**: Modular, no hardcoding


## Setup Guide
```bash
pip install -r requirements.txt
python train.py --help  # View options
python train.py --model YourUNet  # Example execution
```

References

ACDC Dataset

PyTorch HDF5 Tutorial