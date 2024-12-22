import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image, UnidentifiedImageError

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Custom data generator to handle .tif images
def custom_data_generator(directory, batch_size, target_size):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    while True:
        batch_x, batch_y = next(generator)
        valid_images = []
        for img in generator.filenames:
            try:
                image = Image.open(os.path.join(directory, img)).convert('RGB').resize(target_size)
                valid_images.append(np.array(image) / 255.0)
            except UnidentifiedImageError:
                print(f"Skipping file: {img}")
        if len(valid_images) < batch_size:
            continue  # Skip this batch if it doesn't have enough valid images
        batch_x = np.array(valid_images)
        yield batch_x, batch_y[:len(valid_images)]

# Load and preprocess the dataset
def load_data(train_dir, test_dir, batch_size=32, target_size=(64, 64)):
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError("Dataset directories do not exist")

    train_generator = custom_data_generator(train_dir, batch_size, target_size)
    validation_generator = custom_data_generator(test_dir, batch_size, target_size)

    return train_generator, validation_generator

# Train the model
def train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs=25):
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

# Detect pedestrians in an image
def detect_pedestrians(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((64, 64))
    image_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    prediction = model.predict(image_array)
    if prediction[0][0] > 0.5:
        print("Pedestrian detected")
    else:
        print("No pedestrian detected")

if __name__ == "__main__":
    train_dir = 'CVC-14\\Night\\FIR\\Train'
    test_dir = 'CVC-14\\Night\\FIR\\NewTest'
    model = create_model()
    train_generator, validation_generator = load_data(train_dir, test_dir)
    steps_per_epoch = 8000 // 32
    validation_steps = 2000 // 32
    train_model(model, train_generator, validation_generator, steps_per_epoch, validation_steps)
    detect_pedestrians(model, 'images\\2014_05_04_23_14_46_264000.tif')