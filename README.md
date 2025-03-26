# visual-recognition-hw1

# NYCU Computer Vision 2025 Spring HW1

## Student Information
- **Student ID**: 111550203
- **Name**: 提姆西
  
## Introduction
This project focuses on developing a robust biological diversity classification model using deep learning techniques, specifically exploring various ResNet architectures and advanced model modifications. The research aims to demonstrate the effectiveness of transfer learning and advanced regularization techniques in plant species identification.

## How to Install

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

### Installation Steps
1. Clone the repository:
   ```bash
   https://github.com/tvoitekh/visual-recognition-hw1.git
   cd visual-recognition-hw1
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

## Training Workflow
1. Dataset preparation
2. Individual model training
3. Ensemble prediction
4. Performance evaluation

## Reproduction Guidelines
- Set random seed to 42
- Use early stopping (patience: 10 epochs)
- Optional progressive resizing with `--progressive` flag

## Running the Project
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

## Performance Snapshot
### Leaderboard Performance
- **Best Single Model**: ResnNext50 (Advanced)
  - Competition Score: 93%
- **Ensemble Performance**: 92%

<img width="946" alt="image" src="https://github.com/user-attachments/assets/301e33fe-4af7-4138-bf26-ee559dc8a033" />
Leaderboard Snapshot


### Model Comparison
| Model      | Validation Accuracy | F1 Score |
|------------|---------------------|----------|
| ResNet18   | 0.7967                 | 0.7824       |
| ResNet34   | 0.8200                 | 0.8047       |
| ResNet50   | 0.8633                 | 0.8574       |
| ResnNext50 | 0.8800                 | 0.8710     |

  

## PEP8 adherence and Code Linting
<img width="935" alt="image" src="https://github.com/user-attachments/assets/383c1db7-0f5c-40e8-950c-c179a125548c" />
