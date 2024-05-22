import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load the malaria dataset
dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, shuffle_files=True)

# Define constants
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1
BATCH_SIZE = 32
IMG_HEIGHT = 100
IMG_WIDTH = 100
NUM_CLASSES = dataset_info.features['label'].num_classes

len_dataset = dataset_info.splits['train'].num_examples

# Define function to preprocess images
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize pixel values to [0, 1]
    return image, label

# Preprocess and split the dataset
train_dataset = dataset['train'].map(preprocess_image).take(int(TRAIN_RATIO * len_dataset)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = dataset['train'].map(preprocess_image).skip(int(TRAIN_RATIO * len_dataset)).take(int(TEST_RATIO * len_dataset)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset = dataset['train'].map(preprocess_image).skip(int((TRAIN_RATIO + TEST_RATIO) * len_dataset)).take(int(VALIDATION_RATIO * len_dataset)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy:", test_accuracy)
