import cv2
import numpy as np
import pygame
from CarControl import CarControl
import os
import datetime

# Generate a unique folder name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputdir = f"datensatz/recordings_{timestamp}"

if not os.path.exists(outputdir):
    os.makedirs(outputdir)  # Create the directory if it does not exist
    print(f"Created new directory for recording session: {outputdir}")

    #create a csv file to store the control data
    with open(f"{outputdir}/control_data.csv", "w") as f:
        f.write("framecount,timestamp,speed,angle,dir\n")

else:
    print("Directory already exists.")

def save_control_data(framecount, timestamp, speed, angle, dir):
    with open(f"{outputdir}/control_data.csv", "a") as f:
        f.write(f"{framecount},{timestamp},{speed},{angle},{dir}\n")


if not os.path.exists(outputdir):
    print("path not found!!")
framecount = 0
# Globale Variablen für den aktuellen Zustand
current_speed = 0
current_angle = 50

# Fenstergröße entsprechend Ihrem Kamerabild
window_size = (640, 480)
screen = pygame.display.set_mode(window_size)

# Maximale Geschwindigkeit initialisieren
max_speed = 60

def process_image_and_control(frame):
    """
    Diese Funktion verarbeitet das Bild und setzt die Steuerungsdaten.
    """
    global framecount

    framecount += 1
    img_array = np.array(frame)
    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Filter to even out JPEG errors 
    frame = cv2.medianBlur(frame, 3)
    frame = cv2.flip(frame, 1, frame)

    frame = np.rot90(frame, 3)  # Drehen Sie das Bild, falls nötig
    # Save image to /preCalc naming it with <numbering>_<steeringval>_<speedval> as a .png
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")


    filename = f"{framecount}.png"
    save_control_data(framecount, timestamp, current_speed, current_angle, filename)
    print(f'saved to {filename}')
    cv2.imwrite(os.path.join(outputdir, filename), frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.update()

#car_control = CarControl('192.168.1.162', 8000, 'http://192.168.1.162:8000/stream')
#car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')
car_control = CarControl('10.42.0.186', 8000, 'http://10.42.0.186:8000/stream')
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()


try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                # Maximalgeschwindigkeit anpassen
                if joystick.get_button(11):  # Taste nach oben
                    max_speed = min(100, max_speed + 10)
                elif joystick.get_button(12):  # Taste nach unten
                    max_speed = max(0, max_speed - 10)
                

                        

        speed= 0
        angle = 0

        # Vorwärts/Rückwärts fahren
        r2_value = joystick.get_axis(1)  # R2-Taste (Trigger)
        l2_value = joystick.get_axis(4)  # L2-Taste (Trigger)

        if r2_value > 0:
            speed = int(r2_value * max_speed)
        elif l2_value > 0:
            speed = int(-l2_value * max_speed)

        # Lenkung
        axis_value = joystick.get_axis(2)  # Rechter Stick horizontal
        angle = int(axis_value * 100)  # Wert von -100 bis 100
        
        # Überprüfen, ob eine Änderung stattgefunden hat
        if speed != current_speed or angle != current_angle:
            car_control.setControlData(speed, angle)
            current_speed = speed
            current_angle = angle
        
        frame = car_control.read_frame()
        if frame is not None:
            process_image_and_control(frame)
        else:
            print("frame is none!")
except KeyboardInterrupt:
    pass

car_control.close()
pygame.quit()
