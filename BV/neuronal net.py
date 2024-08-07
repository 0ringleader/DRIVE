#neuronal net.py edited 0708 @3PM by Sven
#Code implements a CNN

import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.regularizers import l1_l2
import time
from setVariables import SetVariables, replySetVariables

# Load configuration variables
config = SetVariables('config.ini')
variables = config.get_variables('neuronalnet.py')
learning_rate = variables.get('learning_rate', 0.001)
epochs = variables.get('epochs', 10)
print_console = variables.get('printConsole', True)
batch_size = variables.get('batch_size', 32)
validation_split = variables.get('validation_split', 0.2)
l1_reg = variables.get('l1_reg', 0.0)
l2_reg = variables.get('l2_reg', 0.0)
dropout_rate = variables.get('dropout_rate', 0.5)
data_dir = variables.get('data_dir', "F:\\Einige Dateien\\DRIVE\\PreCalcOut\\datensatz_v2\\datensatz")
xSize = variables.get('xSize', 256)  # Default to 256 if not found
ySize = variables.get('ySize', 192)  # Default to 192 if not found

# Use xSize and ySize when defining input_shape
input_shape = (ySize, xSize, 3)  # Assuming 3 color channels

replySetVariables('neuronalnet.py')

# Function to print debug messages
def dbg_print(message):
    if print_console:
        print(message)

# Global variables for controlling training
PAUSE_TRAINING = False
STOP_TRAINING = False

# Lists to store training history for plotting
train_losses = []
val_losses = []
train_maes = []
val_maes = []
training_logs = []

# Function to pause training
def pause_training():
    global PAUSE_TRAINING
    PAUSE_TRAINING = True
    dbg_print("\nTraining paused. Press Enter to resume.")

# Function to resume training
def resume_training():
    global PAUSE_TRAINING
    PAUSE_TRAINING = False
    dbg_print("\nTraining resumed.")

# Function to stop training
def stop_training():
    global STOP_TRAINING
    STOP_TRAINING = True
    dbg_print("\nTraining stopped.")

# Function to load and preprocess data from multiple folders
def load_behavioral_cloning_dataset(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    images = []
    angles = []

    # Iterate through all subdirectories
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            csv_path = os.path.join(subdir_path, 'control_data.csv')

            try:
                df = pd.read_csv(csv_path)
                dbg_print(f"Processing subfolder: {subdir}")
                dbg_print(f"CSV columns in {subdir}: {df.columns.tolist()}")
            except FileNotFoundError:
                dbg_print(f"Error: CSV file not found in subfolder {subdir}")
                continue

            for index, row in df.iterrows():
                img_filename = row['dir'].strip()
                img_path = os.path.join(subdir_path, img_filename).replace('\\', '/')

                if not os.path.exists(img_path):
                    dbg_print(f"Warning: Image file does not exist at path {img_path}")
                    continue

                try:
                    img = Image.open(img_path)
                    img = np.array(img)
                    if img is not None:
                        images.append(img)
                        angles.append(row['angle'])
                    else:
                        dbg_print(f"Warning: Failed to load image at path {img_path}")
                except IOError as e:
                    dbg_print(f"Warning: IOError when trying to load image at path {img_path}, Error: {e}")

    if len(images) == 0:
        dbg_print("Error: No images were successfully loaded.")
        return [], []

    images = np.array(images, dtype=np.float32) / 255.0
    angles = np.array(angles, dtype=np.float32)

    dbg_print(f"Loaded {len(images)} images successfully from all subfolders")

    return images, angles

# Function to create the model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape,
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu',
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(Conv2D(48, (3, 3), strides=(1, 1), activation='relu',
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(Dense(10, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=[MeanAbsoluteError()])

    return model

# Function to save the model
def save_model(model, path='model.h5'):
    model.save(path)
    dbg_print(f"Model saved to {path}")

# Function to log training progress
def log_training_progress(message, log_file='training_log.txt', metrics=None):
    with open(log_file, 'a') as f:
        f.write(message)
        if metrics is not None:
            f.write(" - " + f"loss: {metrics['loss']:.4f} - mean_absolute_error: {metrics['mean_absolute_error']:.4f} "
                    + f"- val_loss: {metrics['val_loss']:.4f} - val_mean_absolute_error: {metrics['val_mean_absolute_error']:.4f}")
        f.write('\n')
    dbg_print(message)

# Function to load log data
def load_log_data(log_file='training_log.txt'):
    if not os.path.exists(log_file):
        return []

    with open(log_file, 'r') as f:
        training_logs = f.readlines()

    return training_logs

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs, initial_epoch=0, total_time=0.0):
    start_time = time.time()
    history = None
    for epoch in range(initial_epoch, epochs):
        log_training_progress(f"Epoch {epoch + 1}/{epochs} - ", metrics=None)  # Initialize log line

        epoch_start_time = time.time()
        history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val), verbose=1, batch_size=batch_size)
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time

        # Log loss and MAE for plotting
        train_losses.append(history.history['loss'][0])
        val_losses.append(history.history['val_loss'][0])
        train_maes.append(history.history['mean_absolute_error'][0])
        val_maes.append(history.history['val_mean_absolute_error'][0])

        # Prepare metrics dictionary for logging
        metrics = {
            'loss': history.history['loss'][0],
            'mean_absolute_error': history.history['mean_absolute_error'][0],
            'val_loss': history.history['val_loss'][0],
            'val_mean_absolute_error': history.history['val_mean_absolute_error'][0]
        }

        log_message = f"Time elapsed: {epoch_time:.2f}s - Total time elapsed: {total_time:.2f}s"
        log_training_progress(log_message, metrics=metrics)

        if PAUSE_TRAINING:
            input()  # Wait for user input to resume
            resume_training()

        if STOP_TRAINING:
            break

    return history, total_time

def safe_extract(metrics_part, key):
    try:
        return float(metrics_part.split(f'{key}: ')[1].split()[0])
    except (IndexError, ValueError):
        dbg_print(f"Warning: {key} could not be extracted from: '{metrics_part}'.")
        return None

def plot_training_progress(training_logs):
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for log in training_logs:
        if "loss: " in log and "mean_absolute_error: " in log:
            parts = log.split(' - ')
            metrics_part = ' - '.join(parts[1:])  # Join all parts after the first one

            loss = safe_extract(metrics_part, 'loss')
            mae = safe_extract(metrics_part, 'mean_absolute_error')
            val_loss = safe_extract(metrics_part, 'val_loss')
            val_mae = safe_extract(metrics_part, 'val_mean_absolute_error')

            if None not in [loss, mae, val_loss, val_mae]:
                train_losses.append(loss)
                val_losses.append(val_loss)
                train_maes.append(mae)
                val_maes.append(val_mae)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r--', label='Training Loss')
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_maes, 'r--', label='Training MAE')
    plt.plot(epochs, val_maes, 'b-', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Check for GPU availability
    dbg_print("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            dbg_print(e)

    model_path = 'model.h5'
    log_file = 'training_log.txt'

    # Load images and steering data from all subfolders
    images, angles = load_behavioral_cloning_dataset(data_dir)

    if len(images) == 0:
        dbg_print("No images were loaded. Please check the dataset directory and ensure it contains image files.")
        return

    # Example input shape (height, width, channels)
    input_shape = images[0].shape

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=validation_split, random_state=42)

    # Create or load model
    if os.path.exists(model_path):
        model = load_model(model_path)
        dbg_print(f"Loaded model from {model_path}")
    else:
        model = create_model(input_shape)

    # Initialize training logs (clear the log file)
    with open(log_file, 'w') as f:
        f.write('')

    # Training loop
    with tf.device('/GPU:0'):  # This line ensures the model uses the GPU
        history, total_time = train_model(model, X_train, y_train, X_val, y_val, epochs, 0, 0.0)

    # Save model
    save_model(model, model_path)

    # Save training progress
    log_training_progress(f"Training progress saved. Epoch {epochs}/{epochs}.", log_file, metrics=None)

    # Plot accuracy and loss over epochs
    training_logs = load_log_data(log_file)
    plot_training_progress(training_logs)

if __name__ == "__main__":
    main()
