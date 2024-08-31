import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import time
import cProfile
import pstats

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Define hyperparameters
input_size = 50  # Number of distance values
num_classes = 4  # Number of control commands (w, a, s, d)
num_epochs = 100
batch_size = 128
learning_rate = 0.01

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, history_length=10):
        self.data = pd.read_csv(csv_file).to_numpy()  # Convert DataFrame to NumPy array once
        self.history_length = history_length
    
    def __len__(self):
        return len(self.data) - self.history_length + 1
    
    def __getitem__(self, idx):
        history_data = self.data[idx:idx + self.history_length, :5].flatten()
        distances = torch.tensor(history_data, dtype=torch.float32).view(1, -1)
        controls = torch.tensor(self.data[idx + self.history_length - 1, 5:], dtype=torch.float32)
        return {'distances': distances, 'controls': controls}

# Load data
def load_data():
    print("Loading data...")
    full_dataset = CustomDataset('alles/data_track_04_5loops.csv')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("Data loaded.")
    return train_loader, test_loader

# Define the CNN
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(32 * 25, 128)                                                  # Adjust input size according to output size of the conv layers
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        return out

def train_model(model, train_loader, test_loader):
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            distances = batch['distances'].to(device, non_blocking=True)
            controls = batch['controls'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(distances)
            loss = criterion(outputs, controls)  # Use BCEWithLogitsLoss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()    

        # Calculate accuracy on training data
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for batch in train_loader:
                distances = batch['distances'].to(device, non_blocking=True)
                controls = batch['controls'].to(device, non_blocking=True)

                outputs = model(distances)
                predicted = (outputs > 0.5).float()  # Convert logits to binary predictions
                n_correct += (predicted == controls).sum().item()
                n_samples += controls.numel()

            acc = n_correct / n_samples
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100*acc:.2f}%')

    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.2f} seconds')

    # Test the model
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for batch in test_loader:
            distances = batch['distances'].to(device, non_blocking=True)
            controls = batch['controls'].to(device, non_blocking=True)

            outputs = model(distances)
            predicted = (outputs > 0.5).float()  # Convert logits to binary predictions
            n_correct += (predicted == controls).sum().item()
            n_samples += controls.numel()

        acc = n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test samples: {100*acc:.2f}%')

    # Save the state dictionary
    torch.save(model.state_dict(), 'cnn_track_04_5loops_v2.pth')

def main():
    print("Starting main...")
    train_loader, test_loader = load_data()
    model = ConvNet(num_classes).to(device)
    print("Model initialized.")
    train_model(model, train_loader, test_loader)
    print("Training complete.")

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
