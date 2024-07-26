import cv2
import numpy as np
import os

# output_size
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

def custom_sort(file_name):
    # Zerlege den Dateinamen am ersten "_" und konvertiere den Teil vor dem Punkt in eine Zahl
    number_part = file_name.split(".", 1)[0]
    return int(number_part)

# Holen Sie sich eine sortierte Liste aller PNG-Dateien im Ordner
png_files = [file for file in os.listdir(input_dir) if file.endswith('.png')]
png_files.sort(key=custom_sort)

for filename in png_files:
    input_path = os.path.join(input_dir, filename)

    # Bild öffnen
    img = cv2.imread(input_path)
    # Bild auf xSize x ySize skalieren
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_resized = cv2.resize(img_rotated, (xSize, ySize))

    out = process_frame(img_resized)
    # Pfad zum Ausgabebild im entsprechenden Unterordner
    output_subdir = os.path.join(output_dir, os.path.relpath(os.path.dirname(input_path), input_dir))
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, filename)


    # Bild speichern
    cv2.imwrite(output_path, out)
    cv2.imshow('mask auf Bild', out)

    # Wenn 'q' gedrückt wird, wird das Programm beendet
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
