import cv2
import numpy as np
import pygame
from CarControl import CarControl

# Globale Variablen für den Zustand
x_size = 256
y_size = 192
max_speed = 100

def initialize_pygame(window_size):
    """Initialisiert pygame."""
    pygame.init()
    return pygame.display.set_mode(window_size)

def process_image(frame):
    """Verarbeitet das Bild und bereitet es für die Linienerkennung vor."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rot = np.rot90(frame_gray, 3)
    blurred = cv2.GaussianBlur(frame_rot, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(blurred)
    resized = cv2.resize(contrast_img, (y_size, x_size))
    return resized

def define_roi(edges, image_shape):
    """Definiert eine Region of Interest (ROI)."""
    height, width = image_shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (width / 2, 0),
        (width / 2, height),
        (width, height),
        (width, 0),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

def detect_lines(image):
    """Erkennt Linien im Bild."""
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=40, minLineLength=40, maxLineGap=20)

def calculate_steering_angle(lines, image_shape):
    """Berechnet den Lenkwinkel basierend auf den erkannten Linien."""
    if lines is None:
        return None, None
    
    height, width = image_shape[:2]
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        (left_lines if slope < 0 else right_lines).append(line)

    left_avg = np.mean(left_lines, axis=0).flatten() if left_lines else None
    right_avg = np.mean(right_lines, axis=0).flatten() if right_lines else None

    if left_avg is not None and right_avg is not None:
        x1 = (left_avg[0] + right_avg[0]) / 2
        y1 = (left_avg[1] + right_avg[1]) / 2
        x2 = (left_avg[2] + right_avg[2]) / 2
        y2 = (left_avg[3] + right_avg[3]) / 2
    elif left_avg is not None:
        x1, y1, x2, y2 = left_avg
    elif right_avg is not None:
        x1, y1, x2, y2 = right_avg
    else:
        return None, None

    angle_to_horizontal = np.arctan2(y2 - y1, x2 - x1)
    steering_angle = np.degrees(angle_to_horizontal)
    mapped_steering = np.interp(steering_angle, [-55, 55], [-100, 100])
    return steering_angle, mapped_steering

def handle_pygame_events(speed):
    """Verarbeitet die Eingaben von pygame."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, speed

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        speed = min(speed + 10, max_speed)
    elif keys[pygame.K_DOWN]:
        speed = max(speed - 10, 0)
    return True, speed

def main():
    """Hauptfunktion zum Starten des Steuerungssystems."""
    screen = initialize_pygame((x_size, y_size))
    car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')
    speed = 0

    try:
        running = True
        while running:
            running, speed = handle_pygame_events(speed)
            frame = car_control.read_frame()
            if frame is not None:
                processed_frame = process_image(frame)
                edges = cv2.Canny(processed_frame, 80, 150)
                cropped_edges = define_roi(edges, processed_frame.shape)
                lines = detect_lines(cropped_edges)
                steering_angle, mapped_steering = calculate_steering_angle(lines, processed_frame.shape)
                car_control.set_control_data(speed, mapped_steering)
            else:
                print("Kein Frame erhalten!")
    except KeyboardInterrupt:
        pass
    finally:
        car_control.close()
        pygame.quit()

if __name__ == "__main__":
    main()
