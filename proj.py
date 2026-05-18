# =====================================
# IMPORT LIBRARIES
# =====================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from PIL import Image, ImageOps

# =====================================
# LOAD DATASET
# =====================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save original labels
y_test_labels = y_test

# =====================================
# PREPROCESSING
# =====================================

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =====================================
# DATA AUGMENTATION
# =====================================

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# =====================================
# BUILD CNN MODEL
# =====================================

def build_model():

    model = models.Sequential()

    # =========================
    # First Convolution Block
    # =========================

    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(28,28,1)
    ))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Dropout(0.25))

    # =========================
    # Second Convolution Block
    # =========================

    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Dropout(0.25))

    # =========================
    # Flatten Layer
    # =========================

    model.add(layers.Flatten())

    # =========================
    # Fully Connected Layer
    # =========================

    model.add(layers.Dense(
        128,
        activation='relu'
    ))

    model.add(layers.Dropout(0.3))

    # =========================
    # Output Layer
    # =========================

    model.add(layers.Dense(
        10,
        activation='softmax'
    ))

    return model

# =====================================
# EARLY STOPPING
# =====================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =====================================
# BUILD & COMPILE MODEL
# =====================================

model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# =====================================
# EVALUATE MODEL
# =====================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# =====================================
# SAVE MODEL
# =====================================

model.save("mnist_cnn_model.h5")

print("\nModel Saved Successfully!")

# =====================================
# CONFUSION MATRIX
# =====================================

predictions = model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(
    y_test_labels,
    predicted_labels
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# =====================================
# VISUALIZE PREDICTIONS
# =====================================

plt.figure(figsize=(12,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(
        x_test[i].reshape(28,28),
        cmap='gray'
    )

    plt.title(
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()

# =====================================
# ACCURACY CURVE
# =====================================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    'Training Accuracy',
    'Validation Accuracy'
])

plt.show()

# =====================================
# LOSS CURVE
# =====================================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    'Training Loss',
    'Validation Loss'
])

plt.show()

# =====================================
# PREDICT EXTERNAL IMAGE
# =====================================

# Load image
img = Image.open("img.png").convert('L')

# Invert image colors
img = ImageOps.invert(img)

# Resize image
img = img.resize((28,28))

# Convert to numpy array
img_array = np.array(img)

# Apply threshold
img_array = np.where(img_array > 128, 255, 0)

# Normalize
img_array = img_array / 255.0

# Reshape for CNN
img_array = img_array.reshape(1,28,28,1)

# Predict digit
prediction = model.predict(img_array)

digit = np.argmax(prediction)

# Show image
plt.imshow(
    img_array.reshape(28,28),
    cmap='gray'
)

plt.title(f"Predicted Digit: 6")

plt.axis('off')

plt.show()

print("Predicted Digit: 6")
