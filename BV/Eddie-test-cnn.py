import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from cnn9 import ConvNet  # Ensure to replace cnn9 with the correct file name if it's different
import numpy as np
import pandas as pd

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters (ensure these match the ones used in training)
input_height = 192  # Height of images
input_width = 256   # Width of images
num_channels = 1    # Number of channels for grayscale images
num_classes = 1     # Number of control commands (only angle as output)
sequence_length = 10  # Number of past control data points to consider

# Controller Class
class Controller:
    def __init__(self, model_path, device):
        self.device = device
        self.model = ConvNet(num_classes, sequence_length).to(self.device)
        
        # Load the state dict separately to mitigate pickle risks
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),  # Resize image to the desired size
            transforms.ToTensor(),  # Convert image to tensor
        ])

        # Load past control sequence data (this is a simulation for the initial values)
        self.control_sequence = torch.zeros((sequence_length, 1), dtype=torch.float32).to(self.device)

    def predict(self, image_path):
        # Load image
        image = Image.open(image_path).convert('L')  # Use 'L' for grayscale images

        # Apply transformations
        image = self.transform(image)

        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)

        # Predict using the model
        with torch.no_grad():
            output = self.model(image, self.control_sequence.unsqueeze(0))

        # Extract speed and angle from the output
        speed, angle = output[0, 0].item(), output[0, 1].item()

        # Update the control sequence with the latest prediction
        self.control_sequence = torch.roll(self.control_sequence, shifts=-1, dims=0)
        self.control_sequence[-1, 0] = angle

        return speed, angle

    def update_control_sequence(self, new_control_data):
        self.control_sequence = torch.tensor(new_control_data, dtype=torch.float32).view(sequence_length, 1).to(self.device)

def plot_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

def main():
    # Load control data from CSV to get the actual value for comparison
    control_df = pd.read_csv('Data/data/total.csv')

    # Example image path and corresponding index (you'll need to replace these with actual data)
    example_index = 2750 - 1  # Set to 7630 based on image_path
    image_path = 'Data/pro_data/2750.png'

    # Get the corresponding actual control data for the example index
    actual_angle = control_df.loc[example_index, 'angle']
    past_control_data = control_df.loc[max(0, example_index - sequence_length): example_index - 1, 'angle'].values

    # Ensure past_control_data has the right shape and pad if necessary
    if len(past_control_data) < sequence_length:
        padding = np.zeros((sequence_length - len(past_control_data), 1), dtype=np.float32)
        past_control_data = np.vstack((padding, past_control_data[:, None]))
    past_control_data = past_control_data[-sequence_length:]

    # Initialize the controller with the trained model
    controller = Controller(model_path='cnn9_pro_data_model_with_control_seq_ultra.pth', device=device)

    # Update the past control sequence with actual past control data
    controller.update_control_sequence(past_control_data)

    # Predict speed and angle for the input image
    speed, angle = controller.predict(image_path)

    # Show the image
    plot_image(image_path)
    
    print(f"Past Control Data used for Prediction: {past_control_data.flatten()}")
    print(f"Predicted Speed: {speed:.2f}")
    print(f"Predicted Angle: {angle:.2f}")
    print(f"Actual Angle: {actual_angle:.2f}")

if __name__ == "__main__":
    main()
