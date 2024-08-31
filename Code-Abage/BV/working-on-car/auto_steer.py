import cv2
import numpy as np
import pygame
from CarControl import CarControl
from imageProcessor import ImageProcessor

# Globale Variablen für den Zustand
x_size = 256
y_size = 192
max_speed = 100
speed = 0

processor = ImageProcessor(x_size, y_size)
processor.offset = 100

def initialize_pygame(window_size):
    """Initialisiert pygame."""
    pygame.init()
    return pygame.display.set_mode(window_size)

def handle_pygame_events(speed):
    """Verarbeitet die Eingaben von pygame."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, speed

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        speed = min(speed + 10, max_speed)
        print(speed)
    elif keys[pygame.K_DOWN]:
        speed = max(speed - 10, 0)
    return True, speed

def main():
    """Hauptfunktion zum Starten des Steuerungssystems."""
    screen = initialize_pygame((x_size, y_size))
    # car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream') # Raspberry Pi
    car_control = CarControl('127.0.0.1', 8000, 'http://127.0.0.1:8000/stream') # Lokal - Simulator
    speed = 0

    try:
        running = True
        filtered_angle = 0
        while running:
            running, speed = handle_pygame_events(speed)
            frame = car_control.read_frame()
            if frame is not None:
                raw_img = frame
                
                img_rotated = cv2.rotate(raw_img, cv2.ROTATE_180)
                img_resized = cv2.resize(img_rotated, (x_size, y_size))
                processor.preprocess_frame(img_resized) 
                processor.detect_lines()
                processor.get_steering_val(draw_lines=True)
                angle = processor.steering_val
                car_control.setControlData(speed, angle)
                screen.blit(pygame.surfarray.make_surface(cv2.flip(cv2.rotate(processor.raw_image, cv2.ROTATE_90_COUNTERCLOCKWISE),0)), (0, 0))
                running, speed = handle_pygame_events(speed)
                pygame.display.update()
            else:
                print("Kein Frame erhalten!")
    except KeyboardInterrupt or pygame.key.get_pressed()[pygame.K_ESCAPE]:
        pass
    finally:
        car_control.close()
        pygame.quit()

if __name__ == "__main__":
    main()
