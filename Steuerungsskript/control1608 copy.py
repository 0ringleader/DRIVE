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
        print(speed)

    
    return True, speed

def main():
    """Hauptfunktion zum Starten des Steuerungssystems."""
    screen = initialize_pygame((x_size, y_size))
    car_control = CarControl('10.42.0.186', 8000, 'http://10.42.0.186:8000/stream')
    speed = 0

    try:
        running = True
        while running:
            running, speed = handle_pygame_events(speed)
            frame = car_control.read_frame()
            if frame is not None:
                raw_img = frame
                img_resized = cv2.resize(frame, (x_size, y_size))
                img_resized = cv2.flip(img_resized)
                # Bildverarbeitung und Lenkungsberechnung
                processed_image = processor.process_frame(img_resized)
                lines = processor.detect_lines(processed_image)
                line_data = processor.extract_line_data(lines)
                line_data = processor.weight_lines(line_data)
                
                for line in line_data:
                    line.draw(img_resized, 0, 0, 255)
                    
                steering_line = processor.get_steering_line(img_resized, line_data)
                        # zeichne das prozessierte kantenbild auf das bild

                    

                if steering_line is not None:
                    steering_angle = int(-steering_line.calculate_angle())
                    x_error = int(steering_line.middle_x - x_size / 2) * 4
                    left_line, right_line = processor.get_right_and_left_line_mean(line_data)
                    if left_line == right_line:
                        steering_angle = steering_angle - x_error
                    else: 
                        steering_angle = steering_angle + x_error
                    print(f"angle1:  {int(-steering_line.calculate_angle())}, middle_error: {steering_line.middle_x - x_size / 2}* 3")
                    steering_line.draw(img_resized, 255, 0, 0)
                    cv2.putText(img_resized, f"Steering Angle: {steering_line.middle_x:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    car_control.setControlData(speed, steering_angle)
                    img_resized_rotated = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_resized_rotated = pygame.surfarray.make_surface(img_resized_rotated)
                    screen.blit(img_resized_rotated, (0, 0))
                    pygame.display.update()
                handle_pygame_events(speed)



            else:
                print("Kein Frame erhalten!")
    except KeyboardInterrupt:
        pass
    finally:
        car_control.close()
        pygame.quit()

if __name__ == "__main__":
    main()
