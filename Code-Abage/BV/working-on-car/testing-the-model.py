import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from CarControl import CarControl
import threading
import tensorflow as tf



output_image_path = "/home/terrox/Documents/johnnysCode/Alex-ai/processed_image.png"

# ImageProcessor class from your new code
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


# Parameters
xSize = 256
ySize = 192

# Initialize maximum speed
max_speed = 100

def custom_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_error = tf.square(error)
    abs_error = tf.abs(error)
    loss = tf.where(abs_error < 1.0, squared_error, abs_error)
    return tf.reduce_mean(loss)

# Load the trained model

# very smooth on real track
#model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/lane_detection_model.keras"

#not fully trained
#model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/lane_detection_model_ultra_25E.keras"

#best????? works on real track and simu
model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/lane_detection_model_ultra_40E.keras"

# also good
#model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/lane_detection_model_ultra_60E.keras"

#shit
#model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/lane_detection_model_ultra_40E_batch16.keras"
model = load_model(model_path, custom_objects={"custom_loss":custom_loss})

# Shared variable for the latest frame
latest_frame = None
frame_lock = threading.Lock()

def preprocess_image(image):
    processor = ImageProcessor(x_size=300, y_size=224)  # Adjust x_size, y_size based on your setup
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)  # Rotate image by 180 degrees before processing
    preprocessed_image = processor.preprocess_frame(rotated_image)
    normalized_frame = preprocessed_image / 255.0
    return cv2.resize(np.expand_dims(normalized_frame, axis=-1), (80, 26))

steering_scale = 200.0
def process_image_and_control(image, model):
    global steering_scale
    processed_frame = preprocess_image(image)
    steering_angle = model.predict(np.expand_dims(processed_frame, axis=0))[0][0]

    # Scale the steering angle from the normalized prediction
    steering_angle *= steering_scale  # Assuming the original angle range is [-100, 100]

    # Convert the processed frame back to a displayable format
    display_frame = (processed_frame * 255).astype(np.uint8)  # Denormalize

    # Display the processed frame using OpenCV
    cv2.imshow('Processed Frame', display_frame)
    cv2.waitKey(1)
    
    return steering_angle

def frame_capture(car_control):
    global latest_frame
    while True:
        frame = car_control.read_frame()
        if frame is not None:
            with frame_lock:
                latest_frame = frame

# Initialize CarControl
#car_control = CarControl('192.168.12.117', 8000, 'http://192.168.12.117:8000/stream')
car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')
#car_control = CarControl('10.42.0.79', 8000, 'http://10.42.0.79:8000/stream')

# Start the frame capture thread
capture_thread = threading.Thread(target=frame_capture, args=(car_control,))
capture_thread.daemon = True
capture_thread.start()

speed = 40

try:
    running = True
    while running:
        with frame_lock:
            frame = latest_frame
        
        if frame is not None:
            steering_angle = -process_image_and_control(frame, model)
            steering_angle = np.clip(steering_angle, -100, 100)  # Clip the predictions to the expected range
            car_control.setControlData(speed, steering_angle)
            #print(f"Speed: {speed}, Steering Angle: {steering_angle}")
        
        # Key press detection
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
        elif key == ord('a'):
            steering_scale = min(steering_scale + 10, 250)
            print(f"steering scale: {steering_scale}")
        elif key == ord('d'):
            steering_scale = max(steering_scale - 10, 100)
            print(f"steering scale: {steering_scale}")

except KeyboardInterrupt:
    pass

car_control.close()
cv2.destroyAllWindows()
