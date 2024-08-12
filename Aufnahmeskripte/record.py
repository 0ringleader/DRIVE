# record.py edited 1208 @4:40PM by Sven
# Dieses Programm nimmt den Bildschirm und die Controller-Inputs synchronisiert auf

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
import winsound

# Configuration variables
FPS = 50.0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

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
controller = None  # Diese Variable wird den Controller speichern
record_thread = None  # Der Thread für die Aufnahme
current_record_dir = None  # Das aktuelle Verzeichnis für die Aufnahme

# Initialize CSV file
def init_csv(record_dir):
    global csv_file, csv_writer
    csv_path = os.path.join(record_dir, "input_data.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_count", "timestamp", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y", "LT", "RT", "Dpad_Up", "Dpad_Right", "Dpad_Down", "Dpad_Left"])

# Screen recording function
def record_screen():
    global recording, frame_count, current_record_dir, controller
    while recording:
        try:
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_filename = os.path.join(current_record_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)

            if controller:
                controller_inputs = controller.get_controller_inputs()
                timestamp = time.time() - start_time
                csv_writer.writerow([frame_count, f"{timestamp:.3f}"] + controller_inputs)

            frame_count += 1
            time.sleep(1 / FPS)
        except Exception as e:
            print(f"Error during recording: {e}")

# Play a beep sound
def play_beep():
    frequency = 1000  # Set frequency to 1000 Hz
    duration = 200  # Set duration to 200 ms
    winsound.Beep(frequency, duration)

# Toggle recording
def toggle_recording():
    global recording, start_time, frame_count, record_thread, current_record_dir
    if not recording:
        # Start recording
        recording = True
        start_time = time.time()
        frame_count = 0
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        current_record_dir = os.path.join(records_dir, f'record_{timestamp}')
        os.makedirs(current_record_dir, exist_ok=True)
        init_csv(current_record_dir)
        record_thread = threading.Thread(target=record_screen, daemon=True)
        record_thread.start()
        play_beep()  # Beep at start
        print("Recording started...")
    else:
        # Stop recording
        recording = False
        if record_thread:
            record_thread.join()  # Warten Sie auf das Ende des Threads
        if csv_file:
            csv_file.close()
        play_beep()  # Beep at stop
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
