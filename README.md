# 🧠 MNIST Deep Learning Project (CNN)

## 📌 Description
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

The model learns patterns from grayscale images and predicts the correct digit with high accuracy.

---

## 📊 Dataset
- MNIST Dataset (Keras built-in)
- 60,000 training images
- 10,000 testing images
- Image size: 28x28 grayscale

---

## 🧹 Preprocessing
- Normalization (scaling pixel values from 0 to 1)
- Reshaping images to fit CNN input format (28, 28, 1)
- Splitting into training and testing sets (already provided by MNIST)

---

## 🧠 Model Architecture
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling (2x2)
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling (2x2)
- Flatten layer
- Dense layer (128 neurons, ReLU)
- Dropout (0.5) for regularization
- Dense output layer (10 neurons, Softmax)

---

## ⚙️ Experiments
Two different optimizers were tested for comparison:
- Adam Optimizer
- SGD Optimizer

The goal was to analyze the effect of optimizers on model performance.

---

## 📈 Evaluation Metrics
The model was evaluated using:
- Test Accuracy
- Test Loss
- Training vs Validation Accuracy
- Training vs Validation Loss

---

## 📊 Results Comparison

| Model    | Optimizer | Accuracy | Loss |
|----------|----------|----------|------|
| Model A  | Adam     | ~99%     | ~0.03 |
| Model B  | SGD      | ~97%     | ~0.07 |

---

## 🔬 Improvements
- Dropout was used to reduce overfitting
- Two optimizers were compared for performance analysis
- Validation split was used during training
- EarlyStopping was applied to prevent overtraining
- Data Augmentation was used to improve generalization

---

## 🧪 External Image Prediction
The trained model can predict handwritten digits from external images by:
- Converting image to grayscale
- Resizing to 28x28 pixels
- Normalizing pixel values
- Reshaping to match CNN input format

This helps test the model in real-world scenarios.

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow pandas
