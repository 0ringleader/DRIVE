#driveCNN by Sven

import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess images from the stream
def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.resize((256, 192))  # Assuming xSize=256 and ySize=192
    img = img.convert("RGB")
    image = np.array(img)
    return image / 255.0

# Function to make a prediction from the image
def predict_steering_angle(model, image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    predicted_angle = model.predict(processed_image)
    return predicted_angle[0][0]

# Pretend function to control the car with the predicted steering angle
def control_car(steering_angle):
    # This function would include the actual commands to control the car's steering
    print(f"Predicted Steering Angle: {steering_angle}")

def main():
    # Placeholder for video stream and CSV data
    # Adjust below placeholders to actual data sources if available
    video_stream = "path_to_video_stream_or_video_file"
    csv_input = "path_to_controller_input.csv"

    # Read the CSV input for metadata or additional control input information
    control_data = pd.read_csv(csv_input)

    # Initialize video stream
    cap = cv2.VideoCapture(video_stream)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the steering angle from the current frame
        steering_angle = predict_steering_angle(model, frame)
        
        # Control the car based on the predicted angle
        control_car(steering_angle)

        # Display the frame with a visualization of the predicted angle
        # For demonstration purposes (optional)
        cv2.putText(frame, f"Angle: {steering_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Autonomous Driving', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()