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
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix

# =====================================
# LOAD DATASET
# =====================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save labels before one-hot
y_test_labels = y_test

# =====================================
# PREPROCESSING
# =====================================

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
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

datagen.fit(x_train)

# =====================================
# BUILD MODEL
# =====================================

def build_model():

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ))

    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Dropout(0.25))

    # Flatten
    model.add(layers.Flatten())

    # Dense
    model.add(layers.Dense(
        128,
        activation='relu'
    ))

    model.add(layers.Dropout(0.5))

    # Output
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
# EXPERIMENT 1 -> ADAM
# =====================================

model_adam = build_model()

model_adam.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_adam = model_adam.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# Evaluate
adam_loss, adam_acc = model_adam.evaluate(
    x_test,
    y_test
)

# =====================================
# SAVE MODEL
# =====================================

model_adam.save("mnist_cnn_model.h5")

print("Model Saved Successfully!")

# =====================================
# CONFUSION MATRIX
# =====================================

predictions = model_adam.predict(x_test)

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
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================
# PREDICTION VISUALIZATION
# =====================================

plt.figure(figsize=(12,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(
        x_test[i].reshape(28,28),
        cmap='gray'
    )

    plt.title(
        f"Pred: {predicted_labels[i]}"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()

# =====================================
# ACCURACY CURVE
# =====================================

plt.figure(figsize=(10,5))

plt.plot(history_adam.history['accuracy'])
plt.plot(history_adam.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    'Train Accuracy',
    'Validation Accuracy'
])

plt.show()

# =====================================
# LOSS CURVE
# =====================================

plt.figure(figsize=(10,5))

plt.plot(history_adam.history['loss'])
plt.plot(history_adam.history['val_loss'])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    'Train Loss',
    'Validation Loss'
])

plt.show()

# =====================================
# RESULTS TABLE
# =====================================

results = pd.DataFrame({
    "Model": ["CNN + Adam"],
    "Accuracy": [adam_acc],
    "Loss": [adam_loss]
})

print(results)


# =====================================
# PREDICT EXTERNAL IMAGE
# =====================================

from tensorflow.keras.models import load_model
from PIL import Image

# Load Saved Model
model = load_model("mnist_cnn_model.h5")

# Load Image
img = Image.open("images.png").convert('L')

# Resize to 28x28
img = img.resize((28,28))

# Convert image to array
img_array = np.array(img)

# Invert colors
# لأن MNIST الخلفية سوداء والرقم أبيض
img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0

# Reshape for CNN
img_array = img_array.reshape(1,28,28,1)

# Predict
prediction = model.predict(img_array)

# Get predicted digit
digit = np.argmax(prediction)

# Show image
plt.imshow(img_array.reshape(28,28), cmap='gray')

plt.title(f"Predicted Digit: {digit}")

plt.axis('off')

plt.show()

print("Predicted Digit:", digit)اشرحه بقي حرف حرف
