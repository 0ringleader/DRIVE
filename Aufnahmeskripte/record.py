#record.py edited 0608 @11:40PM by Sven
#Dieses Programm nimmt den Bildschirm in einem Video sowie Maus, Tastatur und Controller jeweils in einer .csv Datei im Ordner records, der im selbern Verzeichnis wie record.py liegen muss

#Aktuell funktioniert nur das loggen der Inputs, leider kein Video :(

import pygame
import time
import csv
import cv2
import numpy as np
import threading
import winsound
from datetime import datetime
import pyautogui
from pynput import keyboard, mouse
from queue import Queue
import os
from setVariables import SetVariables

# Konfigurationsvariablen initialisieren
config = SetVariables('config.ini')
variables = config.get_variables('record.py')

# Parameter für Controller und Bildschirmaufzeichnung
FPS = float(variables.get('FPS', 20.0))
SCREEN_WIDTH = int(variables.get('screen_width', pyautogui.size().width))
SCREEN_HEIGHT = int(variables.get('screen_height', pyautogui.size().height))
csv_filename = variables.get('csv_filename', "controller_data.csv").strip('"')
mouse_log_filename = variables.get('mouse_log_filename', "mouse_data.csv").strip('"')
keyboard_log_filename = variables.get('keyboard_log_filename', "keyboard_data.csv").strip('"')
detect_d_pad_inputs = variables.get('detect_d_pad_inputs', True)
plot_controller_inputs = variables.get('plot_controller_inputs', False)
time_window = float(variables.get('time_window', 10.0))
update_interval = float(variables.get('update_interval', 0.05))
sleep_interval = float(variables.get('sleep_interval', 0.01))
log_mouse_inputs = variables.get('log_mouse_inputs', True)
log_keyboard_inputs = variables.get('log_keyboard_inputs', True)
log_controller_inputs = variables.get('log_controller_inputs', True)
print_console = variables.get('printConsole', True)

# Funktion zum Druck auf der Konsole
def dbg_print(message):
    if print_console:
        print(message)

# Debugging-Ausgabe
dbg_print(f"Loaded variables for record.py: {variables}")

# Sicherstellen, dass der records-Ordner existiert
script_dir = os.path.dirname(os.path.abspath(__file__))
records_dir = os.path.join(script_dir, "records")
os.makedirs(records_dir, exist_ok=True)

# Flags und Events für die Synchronisierung
recording = False
recording_event = threading.Event()

# Daten-Queues initialisieren
controller_data_queue = Queue()
screen_data_queue = Queue()
mouse_data_queue = Queue()
keyboard_data_queue = Queue()

# Initialisieren der Dateipfade
controller_file = None
screen_file = None
mouse_file = None
keyboard_file = None

# Initialisieren von Pygame und Joystick, falls Controller-Logging aktiviert ist
pygame.init()
pygame.joystick.init()
joystick = None
if log_controller_inputs:
    if pygame.joystick.get_count() == 0:
        dbg_print("No joystick detected. Please connect an Xbox controller and try again.")
        exit()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()
    dbg_print(f"Joystick: {joystick.get_name()}, Axes: {num_axes}, Buttons: {num_buttons}")

# CSV Initialisierungen
def initialize_csv_logging(filename, header):
    try:
        file_path = os.path.join(records_dir, filename)
        file = open(file_path, 'w', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        file.flush()  # Sicherstellen, dass der Header geschrieben wird
        dbg_print(f"CSV logging initialized for {filename}")
        return csv_writer, file
    except Exception as e:
        dbg_print(f"Error initializing CSV logging for {filename}: {e}")
        return None, None

# Initialisieren der CSV-Dateien
if log_controller_inputs:
    controller_csv_writer, controller_file = initialize_csv_logging(
        f'controller_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',
        ["Timestamp"] + [f"Axis_{i}" for i in range(num_axes)] + [f"Button_{i}" for i in range(num_buttons)]
    )
if log_mouse_inputs:
    mouse_csv_writer, mouse_file = initialize_csv_logging(
        f'mouse_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',
        ["Timestamp", "Action", "X", "Y"]
    )
if log_keyboard_inputs:
    keyboard_csv_writer, keyboard_file = initialize_csv_logging(
        f'keyboard_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',
        ["Timestamp", "Action", "Key"]
    )

# Logging-Funktionen
def log_controller_data():
    global controller_csv_writer
    start_time = time.time()
    dbg_print("Logging controller data...")
    while recording_event.is_set():
        pygame.event.pump()
        axes = [joystick.get_axis(i) for i in range(num_axes)]
        buttons = [joystick.get_button(i) for i in range(num_buttons)]
        if detect_d_pad_inputs:
            hats = list(joystick.get_hat(0))  # Assume a single D-Pad
            axis_buttons = axes + buttons + hats
        else:
            hats = [0, 0]
            axis_buttons = axes + buttons + hats
        timestamp = time.time() - start_time
        controller_csv_writer.writerow([timestamp] + axis_buttons)
        controller_file.flush()
        time.sleep(1 / FPS)
    dbg_print("Stopped logging controller data.")

def log_mouse_data():
    dbg_print("Logging mouse data...")
    while recording_event.is_set():
        while not mouse_data_queue.empty():
            timestamp, action, x, y = mouse_data_queue.get()
            mouse_csv_writer.writerow([timestamp, action, x, y])
        mouse_file.flush()
        time.sleep(sleep_interval)
    dbg_print("Stopped logging mouse data.")

def log_keyboard_data():
    dbg_print("Logging keyboard data...")
    while recording_event.is_set():
        while not keyboard_data_queue.empty():
            timestamp, action, key = keyboard_data_queue.get()
            keyboard_csv_writer.writerow([timestamp, action, key])
        keyboard_file.flush()
        time.sleep(sleep_interval)
    dbg_print("Stopped logging keyboard data.")

def on_mouse_move(x, y):
    if recording_event.is_set():
        timestamp = time.time()
        mouse_data_queue.put((timestamp, "MOVE", x, y))
        dbg_print(f"Mouse move logged at {timestamp}: ({x}, {y})")

def on_mouse_click(x, y, button, pressed):
    if recording_event.is_set():
        timestamp = time.time()
        action = "PRESS" if pressed else "RELEASE"
        mouse_data_queue.put((timestamp, action, x, y))
        dbg_print(f"Mouse click logged at {timestamp}: ({action}, {x}, {y})")

def on_mouse_scroll(x, y, dx, dy):
    if recording_event.is_set():
        timestamp = time.time()
        action = "SCROLL"
        mouse_data_queue.put((timestamp, action, x, y))
        dbg_print(f"Mouse scroll logged at {timestamp}: ({action}, {x}, {y})")

def on_key_press(key):
    if recording_event.is_set():
        timestamp = time.time()
        keyboard_data_queue.put((timestamp, "PRESS", str(key)))
        dbg_print(f"Key press logged at {timestamp}: {str(key)}")

def on_key_release(key):
    if recording_event.is_set():
        timestamp = time.time()
        keyboard_data_queue.put((timestamp, "RELEASE", str(key)))
        dbg_print(f"Key release logged at {timestamp}: {str(key)}")

def record_screen():
    global recording, screen_file, recording_event
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = os.path.join(records_dir, f'screen_capture_{timestamp}.mp4')
    screen_file = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))

    if not screen_file.isOpened():
        dbg_print(f"Error: Unable to open video file for writing: {output_file}")
        return

    recording_event.set()
    recording = True

    # Startton
    winsound.Beep(440, 500)
    dbg_print("Recording screen...")

    try:
        frame_count = 0
        while recording:
            img = pyautogui.screenshot()
            frame = np.array(img)

            # Debug: Druck die Größe und Typ des Frames aus
            dbg_print(f"Frame {frame_count}: {frame.shape}, {frame.dtype}")

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            screen_file.write(frame)
            frame_count += 1
            time.sleep(1 / FPS)
    except Exception as e:
        dbg_print(f"Error during screen recording: {e}")
    finally:
        screen_file.release()
        # Stopp-Ton
        winsound.Beep(880, 500)
        dbg_print("Stopped recording screen.")

def toggle_recording():
    global recording, recording_event
    if not recording_event.is_set():
        # Starte Bildschirmaufzeichnung und Controller-Daten-Logging in separaten Threads
        recording_event.set()  # Ensure recording_event is set before starting threads
        screen_thread = threading.Thread(target=record_screen)
        screen_thread.start()
        if log_controller_inputs:
            controller_thread = threading.Thread(target=log_controller_data)
            controller_thread.start()
        if log_mouse_inputs:
            mouse_thread = threading.Thread(target=log_mouse_data)
            mouse_thread.start()
        if log_keyboard_inputs:
            keyboard_thread = threading.Thread(target=log_keyboard_data)
            keyboard_thread.start()
        dbg_print("Recording started...")
    else:
        recording = False
        recording_event.clear()
        dbg_print("Recording stopped...")

def start_data_collection():
    # Starte Bildschirmaufnahme und Controller-Datensammlung
    toggle_recording()
    keyboard_listener = None
    mouse_listener = None
    if log_mouse_inputs:
        mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_mouse_scroll)
        mouse_listener.start()
    if log_keyboard_inputs:
        keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        keyboard_listener.start()
    return keyboard_listener, mouse_listener

def stop_data_collection(keyboard_listener, mouse_listener):
    toggle_recording()
    if keyboard_listener:
        keyboard_listener.stop()
    if mouse_listener:
        mouse_listener.stop()

if __name__ == "__main__":
    dbg_print("Press ESC to start/stop recording.")
    keyboard_listener, mouse_listener = start_data_collection()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_data_collection(keyboard_listener, mouse_listener)
        if controller_file:
            controller_file.close()
        if mouse_file:
            mouse_file.close()
        if keyboard_file:
            keyboard_file.close()
        dbg_print("All files closed.")
