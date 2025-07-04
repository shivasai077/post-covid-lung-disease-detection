import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os

# Paths to dataset directories
TRAIN_DIR = os.path.join("dataset", "train")
VAL_DIR = os.path.join("dataset", "val")
TEST_DIR = os.path.join("dataset", "test")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,  
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
# Function to build model
def build_model():
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(3, activation="softmax")(x)  # Multi-class classification
    model = Model(inputs=base_model.input, outputs=output)
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Train model function
def train_model():
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # Unfreeze base model for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=7
    )

    # Save model in the recommended format
    model.save("lung_covid_pneumonia_detection_xception_model.keras")  # Corrected format
    return model

if __name__ == "__main__":
    model = train_model()
