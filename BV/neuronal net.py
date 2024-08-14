#neuronal net.py edited 1408 @9:25AM by Sven
#Code implements a CNN

import numpy as np
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import logging
from setVariables import SetVariables, replySetVariables

# Load configuration variables
config = SetVariables('config.ini')
variables = config.get_variables('neuronalnet.py')
learning_rate = variables.get('learning_rate', 0.0005)
epochs = variables.get('epochs', 50)
print_console = variables.get('printConsole', True)
batch_size = variables.get('batch_size', 32)
validation_split = variables.get('validation_split', 0.2)
l1_reg = variables.get('l1_reg', 0.0)
l2_reg = variables.get('l2_reg', 0.0)
dropout_rate = variables.get('dropout_rate', 0.3)
data_dir = variables.get('data_dir', "F:\\Einige Dateien\\DRIVE\\PreCalcOut\\")
xSize = variables.get('xSize', 256)  # Default to 256 if not found
ySize = variables.get('ySize', 192)  # Default to 192 if not found
patience = variables.get('patience', 1000)
data_source = variables.get('data_source', 'real')

# Generate filename with hyperparameters
filename_suffix = f"_lr{learning_rate}_bs{batch_size}_dr{dropout_rate}_pat{patience}_ep{epochs}"
log_file = f'training_log{filename_suffix}.txt'

# Initialize logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Use xSize and ySize when defining input_shape
input_shape = (ySize, xSize, 3)  # Assuming 3 color channels

replySetVariables('neuronalnet.py')

# Function to print debug messages
def dbg_print(message):
    if print_console:
        print(message)

# Function to load and preprocess data from multiple folders
def load_behavioral_cloning_dataset(data_dir, data_source):
    data = []

    if not os.path.exists(data_dir):
        dbg_print(f"Error: The directory {data_dir} does not exist.")
        return pd.DataFrame()

    csv_filename = 'control_data.csv' if data_source == 'real' else 'input_data.csv'

    for subdir, _, _ in os.walk(data_dir):
        csv_path = os.path.join(subdir, csv_filename)
        if not os.path.exists(csv_path):
            dbg_print(f"Warning: CSV file does not exist at path {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df['subdir'] = subdir  # Add the directory for later access

            if data_source == 'real':
                if 'dir' in df.columns:
                    df = df.rename(
                        columns={"framecount": "frame_count", "speed": "speed", "angle": "angle", "dir": "image_filename"})
                else:
                    dbg_print(f"Error: 'dir' column not found in CSV file at {csv_path}")
                    continue
            else:
                df['image_filename'] = df['frame_count'].apply(lambda x: f"frame_{x:05d}.png")

            data.append(df)
        except Exception as e:
            dbg_print(f"Warning: Cannot read CSV file at {csv_path}: {str(e)}")

    if len(data) == 0:
        dbg_print("Error: No data was successfully loaded.")
        return pd.DataFrame()

    data = pd.concat(data, ignore_index=True)
    dbg_print(f"Loaded {len(data)} rows successfully from all CSV files")

    return data

# Function to preprocess a single image
def preprocess_image(img_path):
    with Image.open(img_path) as img:
        img = img.resize((xSize, ySize))
        img = img.convert("RGB")
        image = np.array(img)
    return image / 255.0

# Function to create a dataset from the DataFrame
def create_dataset(df, batch_size, is_training=True):
    def generator():
        while True:
            batch_data = df.sample(n=batch_size, replace=True)
            for _, row in batch_data.iterrows():
                img_filename = row['image_filename'].strip()
                subdir = row['subdir']
                img_path = os.path.join(subdir, img_filename).replace('\\', '/')

                if os.path.exists(img_path):
                    img = preprocess_image(img_path)
                    targets = np.array([row['angle'], row['speed'], row['speed']], dtype=np.float32)  # angle for left_stick_x, speed for acceleration and right_trigger
                    yield img, targets

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32),
                                             output_shapes=((ySize, xSize, 3), (3,)))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Adjusted model to predict three outputs
def create_model(input_shape):
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Conv2D(48, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dropout(dropout_rate),
        Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dense(10, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dense(3)  # Output layer for angle (left_stick_x), speed (acceleration), speed (right_trigger)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=['accuracy'])

    return model

def save_model(model, path='model.h5'):
    model.save(path)
    dbg_print(f"Model saved to {path}")

def log_training_progress(message, log_file, metrics=None):
    if metrics is not None:
        with open(log_file, 'a') as f:
            f.write(f"Epoch {metrics['epoch']:03d}: "
                    f"- loss: {metrics['loss']:.4f} - accuracy: {metrics['accuracy']:.2%} "
                    f"- val_loss: {metrics['val_loss']:.4f} - val_accuracy: {metrics['val_accuracy']:.2%}\n")
    with open(log_file, 'a') as f:
        f.write(message + "\n")
    dbg_print(message)

def load_log_data(log_file):
    if not os.path.exists(log_file):
        return []

    with open(log_file, 'r') as f:
        training_logs = f.readlines()

    return training_logs

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, total_time):
        super().__init__()
        self.log_file = log_file
        self.total_time = total_time

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        }
        epoch_time = time.time() - self.total_time
        self.total_time = time.time()
        log_message = f"Time elapsed for epoch: {epoch_time:.2f}s"
        log_training_progress(log_message, self.log_file, metrics=metrics)

def train_model(model, train_gen, val_gen, train_data_len, val_data_len, epochs, initial_epoch=0, total_time=0.0):
    start_time = time.time()
    steps_per_epoch = train_data_len // batch_size
    validation_steps = val_data_len // batch_size

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    custom_callback = CustomCallback(log_file, start_time)

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps, verbose=1,
                        callbacks=[early_stopping, model_checkpoint, custom_callback])

    total_time += time.time() - start_time

    return history, total_time

def safe_extract(metrics_part, key):
    try:
        import re
        pattern = re.compile(rf"{key}: (?P<value>\d+\.\d+)")
        match = pattern.search(metrics_part)
        if match:
            return float(match.group('value'))
        else:
            raise ValueError("Key not found")
    except (IndexError, ValueError):
        dbg_print(f"Warning: {key} could not be extracted from: '{metrics_part}'.")
        return None

def plot_training_progress(training_logs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for log in training_logs:
        if "loss: " in log and "accuracy: " in log:
            parts = log.split(' - ')
            metrics_part = ' - '.join(parts[1:])  # Join all parts after the first one

            loss = safe_extract(metrics_part, 'loss')
            accuracy = safe_extract(metrics_part, 'accuracy')
            val_loss = safe_extract(metrics_part, 'val_loss')
            val_accuracy = safe_extract(metrics_part, 'val_accuracy')

            if None not in [loss, accuracy, val_loss, val_accuracy]:
                train_losses.append(loss)
                val_losses.append(val_loss)
                train_accuracies.append(accuracy)
                val_accuracies.append(val_accuracy)

    d_train_losses = np.diff(train_losses)
    d_val_losses = np.diff(val_losses)
    d_train_accuracies = np.diff(train_accuracies)
    d_val_accuracies = np.diff(val_accuracies)

    dbg_print(f"Training Losses: {train_losses}")
    dbg_print(f"Validation Losses: {val_losses}")
    dbg_print(f"Training Accuracies: {train_accuracies}")
    dbg_print(f"Validation Accuracies: {val_accuracies}")

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'r--', label='Training Loss')
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, 'r--', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(d_train_losses) + 1), d_train_losses, 'r--', label='Training Loss Derivative')
    plt.plot(range(1, len(d_val_losses) + 1), d_val_losses, 'b-', label='Validation Loss Derivative')
    plt.title('Change in Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Delta Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(d_train_accuracies) + 1), d_train_accuracies, 'r--', label='Training Accuracy Derivative')
    plt.plot(range(1, len(d_val_accuracies) + 1), d_val_accuracies, 'b-', label='Validation Accuracy Derivative')
    plt.title('Change in Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Delta Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot with parameters in the filename
    plt.savefig(f"training_progress{filename_suffix}.png")
    plt.show()

def main():
    dbg_print("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            dbg_print(e)

    model_path = 'model.h5'

    dbg_print("Loading dataset...")
    data = load_behavioral_cloning_dataset(data_dir, data_source)

    if data.empty:
        dbg_print("No data was loaded. Please check the dataset directory and ensure it contains CSV files with image paths.")
        return

    dbg_print("Splitting dataset into train and validation sets...")
    train_data, val_data = train_test_split(data, test_size=validation_split, random_state=42)

    dbg_print(f"Training data: {len(train_data)} samples")
    dbg_print(f"Validation data: {len(val_data)} samples")

    dbg_print("Creating training and validation generators...")
    train_gen = create_dataset(train_data, batch_size, is_training=True)
    val_gen = create_dataset(val_data, batch_size, is_training=False)

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            dbg_print(f"Loaded model from {model_path}")
        except (OSError, ValueError) as e:
            dbg_print(f"Error loading model: {str(e)}")
            dbg_print("Creating a new model instead.")
            model = create_model(input_shape)
    else:
        model = create_model(input_shape)
        model.save(model_path)
        dbg_print(f"Created and saved a new model to {model_path}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=['accuracy'])

    model.summary()

    open(log_file, 'w').close()

    with tf.device('/GPU:0'):
        history, total_time = train_model(model, train_gen, val_gen, len(train_data), len(val_data), epochs, 0, 0.0)

    model.save(model_path)
    dbg_print(f"Model saved to {model_path}")

    training_logs = load_log_data(log_file)
    plot_training_progress(training_logs)

if __name__ == "__main__":
    main()
