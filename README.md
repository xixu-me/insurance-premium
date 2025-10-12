# Insurance Premium Prediction Model

A deep learning model for predicting insurance premiums using PyTorch for Kaggle's [Regression with an Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e12) competition.

## Overview

This project implements a neural network model to predict insurance premium amounts based on various customer features. The model uses a multi-layer perceptron (MLP) architecture with dropout layers for regularization.

## Features

- Multi-layer neural network architecture
- Data preprocessing pipeline for both numerical and categorical features
- Early stopping mechanism to prevent overfitting
- CUDA support for GPU acceleration
- Comprehensive training and validation loops
- Progress tracking with tqdm

## Project Structure

```plaintext
.
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── model/
│   └── best_model.pth
│
└── results/
    └── submission.csv
```

## Requirements

- Python 3.11+
- PyTorch
- pandas
- scikit-learn
- tqdm

## Usage

1. Place your data files in the `data/` directory:
   - `train.csv`: Training data
   - `test.csv`: Test data
   - `sample_submission.csv`: Sample submission format

2. Run the training script `insurance_regression.ipynb`

3. Find predictions in `results/submission.csv`

## Data Preprocessing

The model includes comprehensive preprocessing steps:

- Missing value imputation
- Categorical feature encoding
- Numerical feature standardization
- Feature scaling and normalization

## Hardware Requirements

- Supports both CPU and CUDA GPU execution
- CUDA GPU recommended for faster training

## License

Copyright &copy; [Xi Xu](https://xi-xu.me)

Licensed under the [GPL-3.0](LICENSE) license.  
