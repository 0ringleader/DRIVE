import cv2
import numpy as np
import pygame
from CarControl import CarControl

# Pygame initialisieren
pygame.init()

# Fenstergröße festlegen
xSize = 256
ySize = 192
screen = pygame.display.set_mode((xSize, ySize))

# CarControl initialisieren
car = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')

# Geschwindigkeit und Lenkwinkel
speed = 0.5  # Anfangsgeschwindigkeit (anpassbar)
steering = 0  # Lenkwinkel (wird durch die Linienerkennung bestimmt)

def process_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Erhöhung des Kontrasts mit CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)

    cv2.imshow('contrast_img', contrast_img)
    
    # Kanten mit adaptiven Parametern erkennen
    blurred = cv2.GaussianBlur(contrast_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    
    # Definiere eine Region of Interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    polygon = np.array([[
        (0, height * 1/2),
        (width, height * 1/2),
        (width, height),
        (0, height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges

def detect_lines(image):
    # Hough Transformation zur Linienerkennung
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    return lines

def calculate_steering_angle(image, lines):
    if lines is None:
        return None
    
    height, width, _ = image.shape
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:  # Linie nach links geneigt
            left_lines.append(line)
        else:  # Linie nach rechts geneigt
            right_lines.append(line)
    
    if len(left_lines) > len(right_lines):
        dominant_lines = left_lines
    else:
        dominant_lines = right_lines
    
    if not dominant_lines:
        return None

    x1, y1, x2, y2 = np.mean(dominant_lines, axis=0).flatten()
    angle_to_horizontal = np.arctan2(y2 - y1, x2 - x1)
    steering_angle = np.degrees(angle_to_horizontal)
    
    return steering_angle

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image

running = True
while running:
    # Event-Handling für das Pygame-Fenster
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Frame vom Video-Stream lesen
    frame = car.read_frame()
    if frame is None:
        continue
    
    # Bild skalieren und verarbeiten
    img_rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_resized = cv2.resize(img_rotated, (xSize, ySize))

    processed_image = process_frame(img_resized)
    lines = detect_lines(processed_image)
    steering_angle = calculate_steering_angle(img_resized, lines)
    
    if steering_angle is not None:
        # Den Lenkwinkel für das Auto festlegen
        steering = steering_angle  # Lenkwinkel ohne Anpassung
        car.setControlData(speed, steering)
    
    # Linien auf dem Bild zeichnen
    img_with_lines = draw_lines(img_resized, lines)
    
    # Bild von BGR (OpenCV) zu RGB (Pygame) konvertieren
    img_rgb = cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)
    
    # Das Bild in Pygame anzeigen
    img_surface = pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))
    screen.blit(img_surface, (0, 0))
    pygame.display.update()

# Pygame beenden und Verbindung schließen
car.close()
pygame.quit()
