import cv2
import numpy as np
import os

# Parameters
xSize = 128
ySize = 96

# Verzeichnis für die Eingabebilder
input_dir = "datensatz/datensatz/recordings_20240723_133347_kein_licht_an_kamera1"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut3"

#wenn verzeichnis nicht existiert, wird es erstellt
os.makedirs(output_dir, exist_ok=True)


# wenn die Ordner nicht existieren, werden sie erstellt
os.makedirs(output_dir, exist_ok=True)

def process_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def extract_contrast_regions(diff_image, initial_threshold=50, min_line_length=50, max_line_gap=10):
    # canny edge detection
    edges = cv2.Canny(diff_image, 150, 500)

    # Initialize variables
    lines = None
    threshold = initial_threshold
    max_attempts = 5
    attempts = 0
    mask = np.zeros_like(diff_image)
    filtered_lines = []

    while attempts < max_attempts:
        # Hough Line Transformation
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        if lines is not None:
            filtered_lines.clear()
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Berechne den Neigungswinkel der Linie
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) > 10:
                    filtered_lines.append(line)
                    cv2.line(mask, (x1, y1), (x2, y2), (255), 2)
            
            if filtered_lines:
                # Wenn es gefilterte Linien gibt, breche die Schleife ab
                break

        # Reduziere den Schwellenwert und erhöhe die Max Line Gap für den nächsten Versuch
        threshold = max(1, threshold - 10)
        max_line_gap += 5
        attempts += 1
        print(f"Attempt {attempts}: {len(filtered_lines)} lines found. Threshold: {threshold} | Max Line Gap: {max_line_gap}")

    return mask, len(filtered_lines) if filtered_lines else 0, filtered_lines if filtered_lines else []

def custom_sort(file_name):
    # Zerlege den Dateinamen am ersten "_" und konvertiere den Teil vor dem Punkt in eine Zahl
    number_part = file_name.split(".", 1)[0]
    return int(number_part)

# Holen Sie sich eine sortierte Liste aller PNG-Dateien im Ordner
png_files = [file for file in os.listdir(input_dir) if file.endswith('.png')]
png_files.sort(key=custom_sort)

previous_frame = None

for filename in png_files:
    input_path = os.path.join(input_dir, filename)

    # Bild öffnen
    img = cv2.imread(input_path)
    # Bild auf xSize x ySize skalieren
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_resized = cv2.resize(img_rotated, (xSize, ySize))

    current_frame = process_frame(img_resized)

    if previous_frame is None:
        previous_frame = current_frame
        continue

    # Berechne das Differenzbild
    diff_image = cv2.absdiff(current_frame, previous_frame)
    previous_frame = current_frame

    # Kontrastregionen im Bild extrahieren
    mask, num_lines, contours = extract_contrast_regions(diff_image)

    # Pfad zum Ausgabebild im entsprechenden Unterordner
    output_subdir = os.path.join(output_dir, os.path.relpath(os.path.dirname(input_path), input_dir))
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, filename)

    cv2.imshow('mask', mask)

    maskontop = cv2.addWeighted(img_resized, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # Bild speichern
    cv2.imwrite(output_path, maskontop)
    cv2.imshow('mask auf Bild', maskontop)
    print(f"Bild {filename}: {num_lines} Linien erkannt.")

    # Wenn 'q' gedrückt wird, wird das Programm beendet
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
