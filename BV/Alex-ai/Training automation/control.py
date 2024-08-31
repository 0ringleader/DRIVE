import subprocess
import itertools
import time

# Hyperparameter ranges
batch_sizes = [16, 32, 64]
epochs_list = [25, 30, 40, 50, 60]
learning_rates = [0.0001, 0.001, 0.01]
loss_functions = ['custom_loss', 'huber']

# Function to train a model
def train_model(command):
    subprocess.run(command)

# Loop through every combination of hyperparameters
hyperparameters = list(itertools.product(batch_sizes, epochs_list, learning_rates, loss_functions))

# Train models two at a time
for i in range(0, len(hyperparameters), 2):
    processes = []

    for j in range(2):
        if i + j < len(hyperparameters):
            batch_size, epochs, learning_rate, loss_function = hyperparameters[i + j]
            model_name = f'model_bs{batch_size}_ep{epochs}_lr{learning_rate}_lf{loss_function}'
            model_path = f'C:/Users/Administrator/Documents/AAAAAAAAAAAAAAAAA/BV/auto-train/models2/{model_name}.keras'

            # Train the model with the current hyperparameters
            train_command = [
                'python', 'train.py',
                '--model_path', model_path,
                '--batch_size', str(batch_size),
                '--epochs', str(epochs),
                '--learning_rate', str(learning_rate),
                '--loss_function', loss_function
            ]

            print(f"Training model: {model_name}")
            process = subprocess.Popen(train_command)
            processes.append(process)

    # Wait for both processes to finish before continuing
    for process in processes:
        process.wait()

print("All models have been trained.")
