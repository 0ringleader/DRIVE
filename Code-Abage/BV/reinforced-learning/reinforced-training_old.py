import requests
import cv2
import numpy as np
import threading
import json
from time import sleep
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from CarControl import CarControl  
import random
from collections import deque
import os

# Initialize replay buffer
replay_buffer = deque(maxlen=2000)

# Epsilon-greedy parameters
epsilon = 1.0          # Initial epsilon value
epsilon_min = 0.1      # Minimum epsilon value
epsilon_decay = 0.995  # Epsilon decay rate

ip_address = "192.168.178.32"
#ip_address = "127.0.0.1"

min_lr = 0.00001

def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

def load_existing_model(file_path):
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}")
        return load_model(file_path)
    else:
        print(f"No model found at {file_path}. Creating a new model.")
        return None

def choose_action(model, state):
    global epsilon
    if np.random.rand() <= epsilon:
        return np.random.uniform(-1, 1)  # Random action in the range [-1, 1]
    else:
        return model.predict(np.expand_dims(state, axis=0))[0][0]

    # Decay epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def fetch_road_status():
    try:
        response = requests.get(f"http://{ip_address}:8000/roadStatus")
        status = json.loads(response.text)
        return status['offRoad'], status['failureCount']
    except Exception as e:
        print('Failed to fetch road status:', e)
        return False, 0

def process_image_and_control(frame, model, display_preprocessed=False):
    # Process image using OpenCV
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    height, width, _ = frame.shape
    frame = frame[height//3:, :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reduced_frame = cv2.resize(gray, (80, 26))
    blurred = cv2.GaussianBlur(reduced_frame, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    normalized_frame = edges / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=-1)

    # Predict steering angle with epsilon-greedy strategy
    steering_angle = choose_action(model, input_frame)
    scaled_steering_angle = steering_angle * 100

    if display_preprocessed:
        cv2.imshow('Preprocessed Image', reduced_frame)
        cv2.waitKey(1)

    return scaled_steering_angle, input_frame

def frame_capture(car_control):
    global latest_frame, frame_lock
    while True:
        frame = car_control.read_frame()
        with frame_lock:
            latest_frame = frame

def update_q_values(model, target_model, state, action, reward, next_state, alpha=0.1, gamma=0.9, batch_size=32):
    replay_buffer.append((state, action, reward, next_state))
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    next_q_values = target_model.predict(next_states)
    targets = rewards + gamma * np.max(next_q_values, axis=1)
    q_values = model.predict(states)

    for i in range(batch_size):
        q_values[i][0] = targets[i]

    model.fit(states, q_values, epochs=1, verbose=0)

def update_target_model(model, target_model):
    target_model.set_weights(model.get_weights())

def build_enhanced_model(input_shape):
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Second Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='tanh')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

    return model

def apply_reward_penalty(off_road, reward, penalty, time_on_road, steering_angle):
    if off_road:
        return -penalty
    else:
        steering_penalty = 0.01 * abs(steering_angle)
        return reward + time_on_road * 0.1 - steering_penalty

def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_factor):
    new_lr = max(initial_lr * (lr_decay_factor ** epoch), min_lr)
    optimizer.learning_rate.assign(new_lr)
    return new_lr

if __name__ == "__main__":
    car_control = CarControl(ip_address, 8000, f'http://{ip_address}:8000/stream')
    capture_thread = threading.Thread(target=frame_capture, args=(car_control,))
    capture_thread.daemon = True
    capture_thread.start()

    speed = 50
    frame_lock = threading.Lock()
    latest_frame = None

    model_file = "trained_model.keras"
    model = load_existing_model(model_file) or build_enhanced_model((26, 80, 1))
    target_model = build_enhanced_model((26, 80, 1))
    update_target_model(model, target_model)

    initial_lr = 0.001
    epsilon = 1.0

    reward = 1
    penalty = 30
    total_reward = 0
    time_on_road = 0
    failure_count_s = float('inf')

    update_interval = 2000
    save_interval = 5000
    step_count = 0

    try:
        running = True
        while running:
            with frame_lock:
                frame = latest_frame

            if frame is not None:
                steering_angle, state = process_image_and_control(frame, model)
                car_control.setControlData(speed, steering_angle)
                print(speed, steering_angle)

            off_road, failure_count = fetch_road_status()
            if failure_count > failure_count_s:
                off_road = True
            failure_count_s = failure_count
            print(f"Off road: {off_road}, Failure count: {failure_count}")
            reward_value = apply_reward_penalty(off_road, reward, penalty, time_on_road, steering_angle)
            total_reward += reward_value

            if not off_road:
                time_on_road += 1
            else:
                time_on_road = 0

            if frame is not None:
                next_state = state
                update_q_values(model, target_model, state, steering_angle, reward_value, next_state)

            if off_road:
                print(f"Car left the road! Failure count: {failure_count}. Total reward: {total_reward}")

            step_count += 1
            if step_count % update_interval == 0:
                update_target_model(model, target_model)

            if step_count % save_interval == 0:
                save_model(model, model_file)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                save_model(model, model_file)
            if step_count % update_interval == 0:
                adjust_learning_rate(model.optimizer, step_count // update_interval, initial_lr, 0.9)
                update_target_model(model, target_model)
    except KeyboardInterrupt:
        save_model(model, model_file)
        pass

    car_control.close()
    cv2.destroyAllWindows()