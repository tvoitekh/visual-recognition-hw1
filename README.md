# visual-recognition-hw1

# Plant Classification Project

## Overview
This project focuses on developing a robust plant classification model using deep learning techniques, specifically exploring various ResNet architectures and advanced model modifications.

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU recommended
- Required libraries: 
  ```
  torch
  torchvision
  numpy
  pandas
  matplotlib
  scikit-learn
  tqdm
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-classification.git
   cd plant-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
- Organize your dataset in the following structure:
  ```
  data/
  ├── train/
  │   ├── 0/
  │   ├── 1/
  │   └── ...
  ├── val/
  │   ├── 0/
  │   ├── 1/
  │   └── ...
  └── test/
      └── images/
  ```

## Training

### Single Model Training
Train individual models:
```bash
# Basic ResNet models
python train.py --model resnet18 --epochs 60
python train.py --model resnet34 --epochs 60
python train.py --model resnet50 --epochs 60

# Advanced model with GeM pooling
python train.py --model resnext50 --advanced --epochs 60
```

### Ensemble Prediction
```bash
python ensemble_predict.py
```

This will create submission .csv files for each of the models as well as the ensembling prediction. This way we can observe both individual and ensemble performance.

## Key Results
- Best Single Model: ResnNext50 (Advanced) - 93% Accuracy
- Ensemble Performance: 92% Accuracy

## Reproduction Notes
- Random seed is set to 42 for reproducibility
- Early stopping is implemented with a patience of 10 epochs
- Progressive resizing can be enabled with `--progressive` flag
