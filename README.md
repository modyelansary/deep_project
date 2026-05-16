# deep_project
# 🧠 MNIST Deep Learning Project (CNN)

## 📌 Description
This project uses a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

The model is trained to recognize patterns in grayscale images and predict the correct digit with high accuracy.

---

## 📊 Dataset
- MNIST Dataset (Keras built-in)
- 60,000 training images
- 10,000 testing images
- Image size: 28x28 grayscale

---

## 🧹 Preprocessing
- Normalization (0 → 1 scaling)
- Reshaping for CNN input

---

## 🧠 Model Architecture
- Conv2D layers
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout for regularization

---

## ⚙️ Experiments
Two optimizers were tested:
- Adam
- SGD

---

## 📈 Evaluation Metrics
- Accuracy
- Loss

---

## 📊 Results
- Adam: ~99% accuracy
- SGD: ~97% accuracy

---

## 🚀 How to Run
1. Install requirements:
   pip install tensorflow numpy matplotlib

2. Run notebook or script

3. Train model

---

## 💾 Model
Saved as: mnist_cnn.h5

---

## 🏁 Conclusion
CNN performs very well on MNIST classification and optimizer choice affects performance.
