import cv2
import numpy as np
import pygame
from CarControl import CarControl

# Initialisieren von pygame
pygame.init()



# Fenstergröße entsprechend Ihrem Kamerabild
window_size = (640, 480)
screen = pygame.display.set_mode(window_size)

# Maximale Geschwindigkeit initialisieren

def process_image_and_control(frame):
    """
    Diese Funktion verarbeitet das Bild und setzt die Steuerungsdaten.
    """
    img_array = np.array(frame)
    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Filter to even out JPEG errors 
    frame = cv2.medianBlur(frame, 3)
    frame = cv2.flip(frame, 1, frame)
    frame = np.rot90(frame, 3)  # Drehen Sie das Bild, falls nötig
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.update()

#car_control = CarControl('192.168.1.162', 8000, 'http://192.168.1.162:8000/stream')
car = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')



try:
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        
        frame = car.read_frame()
        process_image_and_control(frame)
except KeyboardInterrupt:
    pass

car.close()
pygame.quit()