# Face Generation and Analysis with AI Models

## 🧠 Project Description

This project uses a subset of the CASIA WebFace dataset to train AI models for facial recognition. Each model is trained on a portion of the dataset to learn how to identify individuals, and is later tested on different images to evaluate its performance.

The aim is to compare the effectiveness of various AI approaches for face identification — both classical (SVM) and deep learning-based (MLP, CNN).

## 🧪 Models Implemented

- **MLP (Multi-Layer Perceptron)**: A fully connected feedforward network used as a baseline.
- **CNN (Convolutional Neural Network)**: Deep learning architecture designed for image classification tasks.
- **SVM (Support Vector Machine)**: A traditional ML model used to classify either raw pixel data or learned embeddings.


## 📂 Dataset

- **CASIA WebFace (subset)** loaded from a preprocessed pickle file:  
  `casia_webface_part_minv.pkl`
- The dataset contains:
  - `images`: Face image tensors
  - `labels`: Identity labels corresponding to the faces


## 🛠️ Technologies Used

- Python 3
- TensorFlow
- NumPy, Pandas
- Pickle (for loading preprocessed datasets)
- Matplotlib (if used for visualization — optional to list here)

