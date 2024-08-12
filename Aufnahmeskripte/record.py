#record.py edited 1208 @11:15AM by Sven
#Dieses Programm nimmt den Bildschirm und die Controller-Inputs synchronisiert auf
#Aktuell funktioniert nur das loggen der Inputs, leider kein Video :(

import cv2
import numpy as np
import pyautogui
import time
import csv
import threading
import os
from datetime import datetime
from pynput import mouse, keyboard
from controller import XboxController  # Importiere die XboxController-Klasse

# Configuration variables
FPS = 20.0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CSV_FILENAME = "input_data.csv"

# Ensure the records folder exists
script_dir = os.path.dirname(os.path.abspath(__file__))
records_dir = os.path.join(script_dir, "records")
os.makedirs(records_dir, exist_ok=True)

# Global variables
recording = False
frame_count = 0
start_time = 0
csv_file = None
csv_writer = None
video_writer = None
controller = None  # Diese Variable wird den Controller speichern
record_thread = None  # Der Thread für die Aufnahme


# Initialize CSV file
def init_csv():
    global csv_file, csv_writer
    csv_path = os.path.join(records_dir, CSV_FILENAME)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_count", "timestamp", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y", "LT", "RT", "Dpad_Up", "Dpad_Right", "Dpad_Down", "Dpad_Left"])


# Screen recording function
def record_screen():
    global recording, frame_count, video_writer, controller
    while recording:
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        frame_count += 1
        if controller:
            controller_inputs = controller.get_controller_inputs()
            timestamp = time.time() - start_time
            csv_writer.writerow([frame_count, f"{timestamp:.3f}"] + controller_inputs)

        time.sleep(1 / FPS)


# Toggle recording
def toggle_recording():
    global recording, start_time, video_writer, frame_count, record_thread
    if not recording:
        # Start recording
        recording = True
        start_time = time.time()
        frame_count = 0
        init_csv()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        video_path = os.path.join(records_dir, f'screen_capture_{timestamp}.mp4')
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
        record_thread = threading.Thread(target=record_screen, daemon=True)
        record_thread.start()
        print("Recording started...")
    else:
        # Stop recording
        recording = False
        if record_thread:
            record_thread.join()  # Warten Sie auf das Ende des Threads
        if video_writer:
            video_writer.release()
        if csv_file:
            csv_file.close()
        print("Recording stopped...")


# Define on_press and on_release accordingly
def on_press(key):
    if key == keyboard.Key.esc:
        toggle_recording()

def on_release(key):
    pass


# Main execution
if __name__ == "__main__":
    print("Press ESC to start/stop recording.")

    # Instantiate the controller
    controller = XboxController(plot_controller_inputs=False)  # Controller ohne Plot initialisieren

    # Set up listeners
    mouse_listener = mouse.Listener(on_move=lambda x, y: None, on_click=lambda *args: None, on_scroll=lambda *args: None)
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    mouse_listener.start()
    keyboard_listener.start()

    # Keep the script running
    keyboard_listener.join()

    # Clean up the controller
    controller.cleanup()

