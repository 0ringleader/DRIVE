import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from CarControl import CarControl
import tensorflow as tf
import requests
import json
import time
import csv
import glob
import logging

logging.basicConfig(level=logging.INFO)

# Define fetch_road_status function with retry mechanism
def fetch_road_status(ip_address, retries=3, delay=1):
    for i in range(retries):
        try:
            response = requests.get(f"http://{ip_address}:8000/roadStatus")
            status = json.loads(response.text)
            return status['offRoad'], status['failureCount']
        except requests.RequestException as e:
            logging.error(f'Failed to fetch road status on attempt {i+1}: {e}')
            time.sleep(delay)
    return False, 0

# ImageProcessor class
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

        # Increasing the line size using dilation
        kernel = np.ones((3, 3), np.uint8)
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

def custom_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_error = tf.square(error)
    abs_error = tf.abs(error)
    loss = tf.where(abs_error < 1.0, squared_error, abs_error)
    return tf.reduce_mean(loss)

def preprocess_image(image):
    processor = ImageProcessor(x_size=300, y_size=224)
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    preprocessed_image = processor.preprocess_frame(rotated_image)
    normalized_frame = preprocessed_image / 255.0
    return cv2.resize(np.expand_dims(normalized_frame, axis=-1), (80, 26))

def process_image_and_control(image, model):
    processed_frame = preprocess_image(image)
    steering_angle = model.predict(np.expand_dims(processed_frame, axis=0))[0][0]

    # Scale the steering angle from the normalized prediction
    steering_angle *= 200  # Assuming the original angle range is [-100, 100]

    # Convert the processed frame back to a displayable format
    display_frame = (processed_frame * 255).astype(np.uint8)  # Denormalize

    # Display the processed frame using OpenCV
    cv2.imshow('Processed Frame', display_frame)
    cv2.waitKey(1)

    return steering_angle

def main():
    # Parameters
    ip_address = '127.0.0.1'
    test_duration = 300  # 5 minutes in seconds
    models_path = './models2/'
    results_csv_path = 'results-training.csv'
    x_size = 256
    y_size = 192

    # Initialize the CSV file
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model_name", "error_count"])

    # Get the list of models to be tested
    model_files = glob.glob(os.path.join(models_path, '*.keras'))

    # Iterate through each model and test it
    for model_file in model_files:
        model_name = os.path.basename(model_file)

        # Add a delay between model loads to avoid any potential conflicts or exhaustion issues
        logging.info(f"Loading model from {model_file} after a brief pause...")
        time.sleep(5)

        # Load the trained model
        print(f"Loading model from {model_file}")
        model = load_model(model_file, custom_objects={"custom_loss": custom_loss})

        # Initialize CarControl
        car_control = CarControl(ip_address, 8000, f'http://{ip_address}:8000/stream')

        # Fetch initial road status
        _, initial_failure_count = fetch_road_status(ip_address)

        speed = 50
        end_time = time.time() + test_duration

        try:
            running = True
            while running and time.time() < end_time:
                frame = car_control.read_frame()

                if frame is not None:
                    steering_angle = -process_image_and_control(frame, model)
                    steering_angle = np.clip(steering_angle, -100, 100)  # Clip the predictions to the expected range
                    car_control.setControlData(speed, steering_angle)

                # Road status update
                _, current_failure_count = fetch_road_status(ip_address)

                # Adding a short delay to avoid overwhelming the server
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            car_control.close()
            cv2.destroyAllWindows()

        # Calculate the final number of errors during the test
        final_failure_count = current_failure_count - initial_failure_count
        print(f"Model {model_name} made {final_failure_count} errors during the test.")

        # Write the results to the CSV file
        with open(results_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, final_failure_count])

if __name__ == "__main__":
    main()