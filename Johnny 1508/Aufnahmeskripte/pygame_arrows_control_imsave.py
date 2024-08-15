import cv2
import numpy as np
import pygame
from CarControl import CarControl
import os
import datetime

# Generate a unique folder name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputdir = f"recordings_{timestamp}"

if not os.path.exists(outputdir):
    os.makedirs(outputdir)  # Create the directory if it does not exist
    print(f"Created new directory for recording session: {outputdir}")
else:
    print("Directory already exists.")


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
max_speed = 100

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

    frame = np.rot90(frame)  # Drehen Sie das Bild, falls nötig
    # Save image to /preCalc naming it with <numbering>_<steeringval>_<speedval> as a .png
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    

    filename = f"{outputdir}/{framecount}_{timestamp}_{current_angle}_{current_speed}.png"
    print(f'saved to {filename}')
    cv2.imwrite(filename, frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.update()

#car_control = CarControl('192.168.1.162', 8000, 'http://192.168.1.162:8000/stream')
car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')



try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Tastenabfragen für Geschwindigkeitsanpassung mit + und -
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    max_speed = min(100, max_speed + 10)  # Erhöhen, maximal 100
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    max_speed = max(0, max_speed - 10)  # Verringern, minimal 0
                
        # Tastenabfragen
        keys = pygame.key.get_pressed()
        speed = 0
        angle = 0  # Neutraler Winkel
        
        if keys[pygame.K_UP]:
            speed = max_speed
        elif keys[pygame.K_DOWN]:
            speed = -max_speed
        
        if keys[pygame.K_LEFT]:
            angle = -100  # -45°
        elif keys[pygame.K_RIGHT]:
            angle = 100  # 45°      
        
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