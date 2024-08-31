import os
import cv2
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import argparse

class line_object:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        if x2 - x1 != 0:
            self.slope = (y2 - y1) / (x2 - x1)
        else:
            self.slope = float('inf')
        self.angle = self.calculate_angle()
        self.middle_x = (x1 + x2) / 2
        self.middle_y = (y1 + y2) / 2
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.weight = 1

    def __str__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}"

    def __repr__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}, Weight: {self.weight}, Left: {self.left}"

    def draw(self, image=None, color=(0, 255, 0)):
        x1_rotated = self.middle_x + (self.middle_x - self.x1)
        y1_rotated = self.middle_y + (self.middle_y - self.y1)
        x2_rotated = self.middle_x + (self.middle_x - self.x2)
        y2_rotated = self.middle_y + (self.middle_y - self.y2)
        cv2.arrowedLine(image, (int(x1_rotated), int(y1_rotated)), (int(x2_rotated), int(y2_rotated)), color, 2, tipLength=0.05)
        cv2.circle(image, (int(self.middle_x), int(self.middle_y)), 5, color, -1)

    def calculate_angle(self):
        x1 = self.x1
        y1 = self.y1
        x2 = self.x2
        y2 = self.y2

        delta_x = x2 - x1
        delta_y = y2 - y1

        theta_rad = np.arctan2(delta_x, delta_y)
        theta_deg = np.degrees(theta_rad)

        angle = theta_deg
        return angle

class ImageProcessor:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.image = None
        self.raw_image = None
        self.hough_lines = None
        self.lines = None
        self.steering_val = 0
        self.extraSteering = 50

    def preprocess_frame(self, image):
        self.raw_image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(blurred)

        edges = cv2.Canny(cl1, 100, 200)

        # increasing the line size using dilation
        kernel = np.ones((3, 3), np.uint8)  # You may adjust the size of the kernel
        better_visible = cv2.dilate(edges, kernel, iterations=1)

        height, width = better_visible.shape
        mask = np.zeros_like(better_visible)

        polygon = np.array([
            [(0, int(height * 1/2)),
            (width, int(height * 1/2)),
            (width, height),
            (0, height)]
        ], np.int32)

        cv2.fillPoly(mask, [polygon], 255)
        cropped_edges = cv2.bitwise_and(better_visible, mask)
        self.image = cropped_edges
        return self.image

# Paths
input_dir = "C:/Users/Administrator/Documents/AAAAAAAAAAAAAAAAA/BV/auto-train/ultra_quadrat/ultra_quadrat"
csv_path = "C:/Users/Administrator/Documents/AAAAAAAAAAAAAAAAA/BV/auto-train/ultra_quadrat/ultra_quadrat.csv"
output_image_path = "/home/terrox/Documents/johnnysCode/Alex-ai/processed_image2.png"

# Argument parser definition
parser = argparse.ArgumentParser(description='Training script with adjustable hyperparameters.')
parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--loss_function', type=str, default='custom_loss', help='Loss function for model training.')

args = parser.parse_args()

# Load CSV data
control_data = pd.read_csv(csv_path)

# Extract relevant columns
filepaths = control_data['dir'].apply(lambda x: os.path.join(input_dir, x)).values
angles = control_data['angle'].values

# Updated preprocess_image function to use ImageProcessor's preprocessing steps
def preprocess_image(image):
    processor = ImageProcessor(x_size=300, y_size=224)  # Adjust x_size, y_size based on your setup
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    preprocessed_image = processor.preprocess_frame(rotated_image)
    normalized_frame = preprocessed_image / 255.0
    output = cv2.resize(normalized_frame, (80, 26))
    return output

# Load and preprocess images
images = []
for filepath in filepaths:
    image = cv2.imread(filepath)
    if image is not None:
        processed_image = preprocess_image(image)
        images.append(np.expand_dims(processed_image, axis=-1))
    else:
        print(f"Warning: Could not read image {filepath}")

images = np.array(images)
angles = np.array(angles) / 100.0  # Normalize angles to range [-1, 1]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2, random_state=42)

def custom_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_error = tf.square(error)
    abs_error = tf.abs(error)
    loss = tf.where(abs_error < 1.0, squared_error, abs_error)
    return tf.reduce_mean(loss)

# Build the model
def build_model(input_shape, learning_rate, loss_function):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['mae'])
    return model

# Choose the loss function dynamically
loss_function_mapping = {
    'custom_loss': custom_loss,
    'huber': Huber()
}

if args.loss_function.lower() not in loss_function_mapping:
    raise ValueError(f"Loss function '{args.loss_function}' is not recognized. Choose from {list(loss_function_mapping.keys())}.")

loss_function = loss_function_mapping[args.loss_function.lower()]

model = build_model((26, 80, 1), args.learning_rate, loss_function)

# Create a callback to save the best model
checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint], verbose=1)

# Predict and print the actual and predicted values
y_pred = model.predict(X_val)
y_pred = np.clip(y_pred * 100.0, -100.0, 100.0)  # Scale predictions back to range [-100, 100] and clip to prevent extreme values
y_val = y_val * 100.0  # Scale validation targets back to range [-100, 100]
for actual, predicted in zip(y_val, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted[0]}")

# Save the final model
model.save(args.model_path)
print(f"Model saved to {args.model_path}")
