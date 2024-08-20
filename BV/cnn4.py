import cProfile
import pstats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import time

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Define hyperparameters
input_height = 120  # Height of images
input_width = 160   # Width of images
num_channels = 1    # Number of channels for grayscale images
num_classes = 2     # Number of control commands (speed and angle)
num_epochs = 50
batch_size = 64     # Adjusted batch size for better memory management
learning_rate = 0.01
epsilon = 10        # Define a threshold for acceptable error

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, control_data, transform=None):
        self.image_paths = image_paths
        self.control_data = np.array(control_data, dtype=np.float32)  # Ensure controls are float32
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('L')  # Use 'L' for grayscale images

        # Load control values
        controls = torch.tensor(self.control_data[idx], dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'controls': controls}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((input_height, input_width)),  # Resize images to the desired size
    transforms.ToTensor(),  # Convert image to tensor
])

def load_data():
    print("Loading data...")

    # Load control data from CSV
    control_df = pd.read_csv('DRIVE/Data/data/control_data.csv')

    # Extract control data (speed and angle) and image paths
    control_data = control_df[['speed', 'angle']].values
    image_paths = control_df['dir'].apply(lambda x: os.path.join('DRIVE/Data/processed_images', x)).tolist()

    # Ensure images and control data match
    assert len(image_paths) == len(control_data), "Mismatch between number of images and control data"

    # Shuffle and split the dataset
    indices = list(range(len(image_paths)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    test_image_paths = [image_paths[i] for i in test_indices]
    train_control_data = [control_data[i] for i in train_indices]
    test_control_data = [control_data[i] for i in test_indices]
    
    # Create datasets and loaders
    train_dataset = CustomDataset(train_image_paths, train_control_data, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_control_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Data loaded.")
    return train_loader, test_loader

# Model definition
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        # Calculate flattened size after convolutions and pooling
        self.flattened_size = self._get_flattened_size(input_width, input_height)
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(256, num_classes)
        self.tanh = nn.Tanh()

    def _get_flattened_size(self, width, height):
        x = torch.zeros(1, num_channels, height, width)  # Example input tensor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool(x)
        
        return x.numel()  # Flattened size

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool(out)
        
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        out = self.tanh(out) * 100  # Scale the output to the range [-100, 100]
        return out

def train_model(model, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    start_time = time.time()  # Start total training timer
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()  # Start timer for this epoch

        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            controls = batch['controls'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, controls)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Calculate number of correct predictions
            absolute_errors = torch.abs(outputs - controls)
            correct = torch.sum(absolute_errors <= epsilon).item()
            correct_predictions += correct
            total_samples += controls.numel()

        scheduler.step()

        # Calculate average loss and accuracy for this epoch
        avg_loss = epoch_loss / len(train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        
        epoch_duration = time.time() - epoch_start_time  # Time taken for this epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Epoch Time: {epoch_duration:.2f} seconds')

    total_training_time = time.time() - start_time  # Total training time
    print(f'Total training time: {total_training_time:.2f} seconds')

    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct_predictions = 0
        total_samples = 0
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            controls = batch['controls'].to(device, non_blocking=True)

            outputs = model(images)
            test_loss += criterion(outputs, controls).item()
            
            # Calculate number of correct predictions
            absolute_errors = torch.abs(outputs - controls)
            correct = torch.sum(absolute_errors <= epsilon).item()
            correct_predictions += correct
            total_samples += controls.numel()

        avg_test_loss = test_loss / len(test_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

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