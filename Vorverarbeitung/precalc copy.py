import cv2
import numpy as np
import os

# Parameters
xSize = 256
ySize = 192

# Verzeichnis für die Eingabebilder
input_dir = "datensatz"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut"

# wenn die Ordner nicht existieren, werden sie erstellt
os.makedirs(output_dir, exist_ok=True)

def extract_contrast_regions(image, initial_threshold=50, min_line_length=40, max_line_gap=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Verwende adaptive Schwellenwertbestimmung
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 3)
    
    equalized = cv2.equalizeHist(gray)

    # feine strukturen abmildern
    kernel = np.ones((3, 3), np.uint8)
    equalized = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel, iterations=1)

    #rauschen entfernen
    equalized = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)

    # canny edge detection
    edges = cv2.Canny(equalized, 150, 500)

    # Initialize variables
    lines = None
    threshold = initial_threshold
    max_attempts = 8
    attempts = 0
    mask = np.zeros_like(gray)
    filtered_lines = []
    cv2.imshow('edges', edges)

    while attempts < max_attempts:
        # Hough Line Transformation
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        if lines is not None:
            filtered_lines.clear()
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Berechne den Neigungswinkel der Linie
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) > 14 and abs(angle) < 76:
                    filtered_lines.append(line)
                    cv2.line(mask, (x1, y1), (x2, y2), (255), 2)
            
            if filtered_lines:
                # Wenn es gefilterte Linien gibt, breche die Schleife ab
                break
        
        # Reduziere den Schwellenwert für den nächsten Versuch
        threshold = max(1, int(threshold / 1.5))
        max_line_gap += 3
        min_line_length -= 5
        attempts += 1
        print(f"Attempt {attempts}: {len(filtered_lines)} lines found. Threshold: {threshold} | Max Line Gap: {max_line_gap}")

    return mask, len(filtered_lines) if filtered_lines else 0, filtered_lines if filtered_lines else []

# Durchlaufe alle Ordner im Eingabeverzeichnis
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Pfad zum Eingabebild
            input_path = os.path.join(root, filename)

            # Bild öffnen
            img = cv2.imread(input_path)
            # Bild auf xSize x ySize skalieren
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_resized = cv2.resize(img_rotated, (xSize, ySize))

            # Kontrastregionen im Bild extrahieren
            mask, num_lines, contours = extract_contrast_regions(img_resized)

            # Pfad zum Ausgabebild im entsprechenden Unterordner
            output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, filename)

            cv2.imshow('mask', mask)

            # Konturen in zweitem Fenster anzeigen
            # cv2.imshow('contours', cv2.drawContours(img_resized, contours, -1, (0, 255, 0), 2))

            maskontop = cv2.addWeighted(img_resized, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            # Bild speichern
            cv2.imwrite(output_path, maskontop)
            cv2.imshow('mask auf Bild', maskontop)
            print(f"Bild {filename}: {num_lines} Linien erkannt.")
            # cv2.waitKey(0)

            # Wenn 'q' gedrückt wird, wird das Programm beendet
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
