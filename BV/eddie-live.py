import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from CarControl import CarControl
import threading

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model parameters
input_height = 192
input_width = 256
num_channels = 1
num_classes = 2
sequence_length = 10

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((input_height, input_width)),  # Resize images to the desired size
    transforms.ToTensor(),  # Convert image to tensor
])

# Model definition
class ConvNet(nn.Module):
    def __init__(self, num_classes, sequence_length):
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

        self.flattened_size = self._get_flattened_size(input_width, input_height)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

        self.control_fc = nn.Linear(sequence_length * 2, 64)
        self.bn_control = nn.BatchNorm1d(64)
        self.relu_control = nn.ReLU()

        self.fc2 = nn.Linear(256 + 64, num_classes)
        self.tanh = nn.Tanh()

    def _get_flattened_size(self, width, height):
        x = torch.zeros(1, num_channels, height, width)
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

        return x.numel()

    def forward(self, x, control_seq):
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

        control_seq = control_seq.view(control_seq.size(0), -1)
        control_out = self.control_fc(control_seq)
        control_out = self.bn_control(control_out)
        control_out = self.relu_control(control_out)

        combined_out = torch.cat((out, control_out), dim=1)

        combined_out = self.fc2(combined_out)
        combined_out = self.tanh(combined_out) * 100

        return combined_out

model = ConvNet(num_classes, sequence_length).to(device)
model.load_state_dict(torch.load('DRIVE/model_with_control_seq.pth'))
model.eval()

# Contains the past control values
past_controls = []

# Shared variable for the latest frame
latest_frame = None
frame_lock = threading.Lock()

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(frame)
    tensor_image = transform(pil_image)
    tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
    return tensor_image

def process_image_and_control(image):
    global past_controls

    processed_frame = preprocess_frame(image)
    past_controls_tensor = torch.tensor([past_controls], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(processed_frame.to(device), past_controls_tensor)

    control_values = outputs.cpu().numpy()[0]
    speed, steering_angle = control_values

    if len(past_controls) >= 10:
        past_controls.pop(0)
    past_controls.append([speed, steering_angle])

    return speed, steering_angle

def frame_capture(car_control):
    global latest_frame
    while True:
        frame = car_control.read_frame()
        if frame is not None:
            with frame_lock:
                latest_frame = frame

# Initialize CarControl
car_control = CarControl('192.168.178.32', 8000, 'http://192.168.178.32:8000/stream')

# Start the frame capture thread
capture_thread = threading.Thread(target=frame_capture, args=(car_control,))
capture_thread.daemon = True
capture_thread.start()

try:
    speed = 50
    running = True
    while running:
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            speed, steering_angle = process_image_and_control(frame)
            car_control.setControlData(speed, steering_angle)
            print(f"Speed: {speed}, Steering Angle: {steering_angle}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            car_control.setControlData(0, steering_angle)
            running = False
        elif key == ord('w'):
            speed = min(speed + 10, 100)
            print(f"Speed: {speed}")
        elif key == ord('s'):
            speed = max(speed - 10, 0)
            print(f"Speed: {speed}")

except KeyboardInterrupt:
    pass

car_control.close()
cv2.destroyAllWindows()
