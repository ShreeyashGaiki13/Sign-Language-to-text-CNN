import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Define the model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(11, activation='softmax'))  # 11 classes: 0-9 and a blank symbol

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Define paths to the training and testing datasets
train_dir = "D:\\Mini Project\\modelData\\train"
test_dir = "D:\\Mini Project\\modelData\\test"

# Load and preprocess images
def load_images(directory):
    images = []
    labels = []
    label_to_int = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'blank': 10}
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(label_to_int.get(label))
    return np.array(images), np.array(labels)

# Load training and testing images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)

# Reshape images to add channel dimension
train_images = train_images.reshape(-1, 48, 48, 1)
test_images = test_images.reshape(-1, 48, 48, 1)

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes=11)
test_labels = to_categorical(test_labels, num_classes=11)

# Train the model
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate model on test set
loss, accuracy = model.evaluate(test_images, test_labels)
print("Loss on test set:", loss)
print("Accuracy on test set:", accuracy)

# Save the model
model.save("sign_language_model.h5")
print("Model saved as 'sign_language_model.h5'")

# Additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on test set
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(test_labels, axis=1), predicted_classes))
print("Classification Report:")
print(classification_report(np.argmax(test_labels, axis=1), predicted_classes))
