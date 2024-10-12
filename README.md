# Speech Emotion Recognition (SER)

This repository contains a project on **Speech Emotion Recognition (SER)** using machine learning techniques. The objective of this project is to classify emotions embedded in speech signals. We use the **Crema dataset** for training and testing the model, and employ **PyTorch** for model development.

## Project Overview

Speech is one of the most natural ways humans express emotions. The goal of this project is to develop a system capable of detecting emotions such as happiness, sadness, anger, and more from speech signals. The project follows a structured approach from data loading to model training and evaluation.

## Steps in the Project

### 1. Data Loading and Preprocessing
- **Dataset**: The project uses the **Crema dataset**, which contains labeled speech data for various emotions.
- **Libraries**: `librosa` is used for audio processing and feature extraction, and `sklearn` for splitting data and standardizing features.
- **Preprocessing**:
  - Audio files are loaded using `librosa`.
  - Features like Mel-frequency cepstral coefficients (MFCCs) are extracted.
  - Labels are encoded for classification.

### 2. Train-Test Split
- The dataset is split into training and testing sets using `train_test_split` from `sklearn`.

### 3. Model Building
- The model is built using **PyTorch** with a neural network architecture. Key components include:
  - Fully connected layers
  - Dropout for regularization
  - Activation functions like ReLU
  - Softmax output for classification

### 4. Model Training
- The model is trained using backpropagation and the Adam optimizer.
- The training process includes loss calculation and optimization for multiple epochs.

### 5. Evaluation
- The model's performance is evaluated using metrics like accuracy, F1 score, and confusion matrix, computed with `sklearn`.
- A detailed classification report and confusion matrix are generated to visualize performance.

### 6. Visualization
- The results are visualized using `matplotlib` and `seaborn` to plot confusion matrices and other relevant figures.

## Libraries and Tools
- `librosa`: For audio processing
- `scikit-learn`: For data splitting, preprocessing, and evaluation
- `matplotlib` and `seaborn`: For visualization
- `TensorFlow` and `PyTorch`: For building and training the deep learning models
