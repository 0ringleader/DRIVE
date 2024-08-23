import cv2
import numpy as np
import pygame
import sys
import os
import time
import threading
import tensorflow as tf
from configparser import ConfigParser
from controller import XboxController
from pynput import keyboard
import pyautogui
import csv
from datetime import datetime
import winsound
import vgamepad as vg

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration variables
config = ConfigParser()
config.read('config.ini')

# Helper function to get configuration variables
def get_bool(section, var_name, default=None):
    try:
        return config.getboolean(section, var_name, fallback=default)
    except ValueError:
        logging.warning(f"The value for {var_name} in {section} cannot be converted to boolean. Using default value {default}.")
        return default

def get_float(section, var_name, default=None):
    try:
        return config.getfloat(section, var_name, fallback=default)
    except ValueError:
        logging.warning(f"The value for {var_name} in {section} cannot be converted to float. Using default value {default}.")
        return default

def get_int(section, var_name, default=None):
    try:
        return config.getint(section, var_name, fallback=default)
    except ValueError:
        logging.warning(f"The value for {var_name} in {section} cannot be converted to int. Using default value {default}.")
        return default

def get_tuple(section, var_name, default=None):
    value = config.get(section, var_name, fallback=default)
    if isinstance(value, tuple):
        return value
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        logging.warning(f"The value for {var_name} in {section} cannot be converted to tuple of int. Using default value {default}.")
        return default

# Get variables from the configuration file and convert
xSize = get_int('precalc', 'xsize', 256)
ySize = get_int('precalc', 'ysize', 192)
clahe_clipLimit = get_float('precalc', 'clahe_cliplimit', 3.0)
clahe_tileGridSize = get_tuple('precalc', 'clahe_tilegridsize', (8, 8))
gaussian_blur_ksize = get_int('precalc', 'gaussian_blur_ksize', 5)
denoising_h = get_float('precalc', 'denoising_h', 30)
denoising_template_window_size = get_int('precalc', 'denoising_template_window_size', 7)
denoising_search_window_size = get_int('precalc', 'denoising_search_window_size', 21)
adaptive_thresh_block_size = get_int('precalc', 'adaptive_thresh_block_size', 11)
adaptive_thresh_C = get_float('precalc', 'adaptive_thresh_C', 2)
rotate = get_bool('precalc', 'rotate', True)
apply_filter = get_bool('precalc', 'apply_filter', True)
apply_canny = get_bool('precalc', 'apply_canny', True)
draw_center_line = get_bool('precalc', 'drawcenterline', True)
centerline_sensitivity = get_float('precalc', 'centerline_sensitivity', 1.0)
drive_delay = get_float('simulation', 'drive_delay', 0.1)
steering_damping_factor = get_float('simulation', 'steering_damping_factor', 0.5)  # New damping factor
mirror_steering = get_bool('simulation', 'mirrorSteering', False)  # New mirror steering option

# Define the correct model path
model_path = 'model.h5'

# Load the model without compiling it
model = tf.keras.models.load_model(model_path, compile=False)

# Compile the model manually with appropriate loss and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Pygame initialization
pygame.init()
pygame.joystick.init()

# Create a Pygame window
screen = pygame.display.set_mode((xSize, ySize))
pygame.display.set_caption('CNN Drive Simulation')

# Function to adjust contrast and brightness
def adjust_contrast_brightness(image_gray, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image_gray, alpha=alpha, beta=beta)

# Function to apply CLAHE
def apply_clahe(image_gray):
    clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
    return clahe.apply(image_gray)

# Function to reduce noise
def reduce_noise(image_gray):
    ksize = (gaussian_blur_ksize, gaussian_blur_ksize)
    blurred = cv2.GaussianBlur(image_gray, ksize, 0)
    denoised = cv2.fastNlMeansDenoising(blurred, None, denoising_h, denoising_template_window_size,
                                        denoising_search_window_size)
    return denoised

# Function to binarize the image
def binarize_image(image_gray):
    _, binary_global = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            adaptive_thresh_block_size, adaptive_thresh_C)
    binary_combined = cv2.bitwise_and(binary_global, binary_adaptive)
    return binary_combined

# Function to detect edges using Canny
def apply_canny_edge_detection(image_binary):
    return cv2.Canny(image_binary, 100, 200)

# Function to crop the image
def crop_image(image):
    height, width = image.shape[:2]
    cropped = image[height // 2:, :]
    return cropped

# Process image through pipeline
def process_image_pipeline(image, steps):
    for step in steps:
        image = step(image)
    return image

# Function to process a single image
def process_single_image(image):
    img_resized = cv2.resize(image, (xSize, ySize))

    if rotate:
        img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    steps = [adjust_contrast_brightness, apply_clahe]
    if apply_filter:
        steps.append(reduce_noise)
    steps.append(binarize_image)
    if apply_canny:
        steps.append(apply_canny_edge_detection)

    img_processed = process_image_pipeline(img_gray, steps)

    # Ensure the processed image has 3 channels by duplicating the grayscale image across the 3 color channels
    img_processed_color = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
    return img_processed_color

def get_screenshot():
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def capture_controller_inputs():
    global controller_inputs
    while running:
        controller_inputs = controller.get_controller_inputs()
        time.sleep(drive_delay)

def on_press(key):
    global running, start_time, frame_count, record_thread, current_record_dir, csv_file, csv_writer
    if key == keyboard.Key.esc:
        if not running:
            running = True
            start_time = time.time()
            frame_count = 0
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            current_record_dir = os.path.join(records_dir, f'record_{timestamp}')
            os.makedirs(current_record_dir, exist_ok=True)
            csv_path = os.path.join(current_record_dir, "input_data.csv")
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["frame_count", "timestamp", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y", "LT",
                 "RT", "Dpad_Up", "Dpad_Right", "Dpad_Down", "Dpad_Left"])
            record_thread = threading.Thread(target=capture_controller_inputs, daemon=True)
            record_thread.start()
            winsound.Beep(1000, 200)  # Beep at start
            print("Simulation started...")
        else:
            running = False
            csv_file.close()
            winsound.Beep(1000, 200)  # Beep at stop
            print("Simulation stopped...")

def on_release(key):
    pass

# Function to control vehicle with the predicted steering angle
def control_vehicle_with_steering_angle(steering_angle, gamepad):
    # Apply mirror steering option
    if mirror_steering:
        steering_angle = -steering_angle

    # Apply damping factor to the steering angle
    damped_steering_angle = steering_angle * steering_damping_factor

    # Convert the steering angle to the suitable input for the controller
    # Assuming that -1 to 1 represents the full range of the joystick
    left_stick_x = damped_steering_angle / 100.0  # Map -100 to 100 range to -1 to 1 range

    # Ensure the value is within the range
    left_stick_x = np.clip(left_stick_x, -1, 1)

    # Convert to vgamepad range (-32768 to 32767)
    vgamepad_value = int(left_stick_x * 32767)

    # Apply the steering using vgamepad
    gamepad.left_joystick(x_value=vgamepad_value, y_value=0)
    gamepad.update()

# Main loop
if __name__ == "__main__":
    controller = XboxController(plot_controller_inputs=False)
    gamepad = vg.VX360Gamepad()

    # Make sure records directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    records_dir = os.path.join(script_dir, "records")
    os.makedirs(records_dir, exist_ok=True)

    running = False
    controller_inputs = [0] * 8  # Initialize with zeros for controller inputs

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if running:
            frame = get_screenshot()
            processed_image = process_single_image(frame)
            processed_image_resized = cv2.resize(processed_image, (xSize, ySize))
            processed_image_expanded = np.expand_dims(processed_image_resized, axis=0)  # Add batch dimension

            # Get predicted steering angle from the model
            steering_angle = model.predict(processed_image_expanded)[0]

            # Clip the steering angle to be within the range -100 to 100
            steering_angle = np.clip(steering_angle, -100, 100)

            # Control the vehicle using the predicted steering angle
            control_vehicle_with_steering_angle(steering_angle, gamepad)

            # Save the processed frame
            frame_filename = os.path.join(current_record_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)

            # Log controller inputs and image data
            timestamp = time.time() - start_time
            csv_writer.writerow([frame_count, f"{timestamp:.3f}"] + controller_inputs)
            frame_count += 1

            # Process the steering_angle as needed, e.g., map to joystick input or directly control motors
            print(f"Predicted Steering Angle: {steering_angle}")

            processed_image_surface = pygame.surfarray.make_surface(processed_image_resized)
            screen.blit(processed_image_surface, (0, 0))
            pygame.display.update()

            pygame.time.delay(int(drive_delay * 1000))  # Adjust delay to maintain the desired FPS

    controller.cleanup()
