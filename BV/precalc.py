#precalc.py edited 0708 @3PM by Sven

import cv2
import numpy as np
import os
import csv
import shutil
from setVariables import SetVariables, replySetVariables

# Laden der Konfigurationsvariablen
config = SetVariables('config.ini')
variables = config.get_variables('precalc.py')

# Hilfsfunktionen zur Konvertierung und Typprüfung
def get_float(var_name, default=None):
    try:
        return float(variables.get(var_name, default))
    except ValueError:
        print(f"Warning: The value for {var_name} cannot be converted to float. Using default value {default}.")
        return default

def get_int(var_name, default=None):
    try:
        return int(variables.get(var_name, default))
    except ValueError:
        print(f"Warning: The value for {var_name} cannot be converted to int. Using default value {default}.")
        return default

def get_tuple(var_name, default=None):
    value = variables.get(var_name, default)
    if isinstance(value, tuple):
        return value
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        print(f"Warning: The value for {var_name} cannot be converted to tuple of int. Using default value {default}.")
        return default

# Variablen aus der Konfigurationsdatei holen und konvertieren
xSize = get_int('xSize', 256)  # Standardwert 256 falls nicht in der config.ini
ySize = get_int('ySize', 192)  # Standardwert 192 falls nicht in der config.ini
edge_strength = get_float('edge_strength', 1.0)  # Neue Variable für die Kantenstärke
noise_h = get_float('noise_h', 10)  # Parameter für Rauschreduzierung
noise_hColor = get_float('noise_hColor', 10)  # Parameter für Rauschreduzierung
noise_templateWindowSize = get_int('noise_templateWindowSize', 7)  # Parameter für Rauschreduzierung
noise_searchWindowSize = get_int('noise_searchWindowSize', 21)  # Parameter für Rauschreduzierung
canny_threshold1 = get_float('canny_threshold1', 50)  # Parameter für Canny-Kantenerkennung
canny_threshold2 = get_float('canny_threshold2', 150)  # Parameter für Canny-Kantenerkennung
clahe_clipLimit = get_float('clahe_clipLimit', 3.0)  # Parameter für Kontrastverbesserung
clahe_tileGridSize = get_tuple('clahe_tileGridSize', (8, 8))  # Parameter für Kontrastverbesserung

# Eingabe- und Ausgabe-Verzeichnisse
input_dir = "PreCalcIn"
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

# Antwort auf gesetzte Variablen ausgeben
replySetVariables('precalc.py')

# Funktionen zur Bildverarbeitung
def reduce_artifacts(edges):
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    return edges

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, noise_h, noise_hColor, noise_templateWindowSize, noise_searchWindowSize)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def extract_and_enhance_edges(image):
    image = reduce_noise(image)
    image = enhance_contrast(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=canny_threshold1, threshold2=canny_threshold2)
    edges = reduce_artifacts(edges)
    return edges

def apply_edge_overlay(image, edges):
    edge_color = (0, 255, 0)  # Grüne Farbe für die Kanten
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color_edges[np.where((color_edges == [255, 255, 255]).all(axis=2))] = edge_color
    return cv2.addWeighted(image, 1, color_edges, edge_strength, 0)

def rotate_image(image):
    # Bild um 90 Grad im Uhrzeigersinn drehen
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def process_images_and_csv(input_dir, output_dir):
    total_folders = sum(1 for _, dirnames, _ in os.walk(input_dir) if not dirnames)
    processed_folders = 0

    for root, _, files in os.walk(input_dir):
        csv_file = next((f for f in files if f.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(root, csv_file)
            relative_path = os.path.relpath(root, input_dir)
            output_path_dir = os.path.join(output_dir, relative_path)
            os.makedirs(output_path_dir, exist_ok=True)

            # CSV-Datei ins Ausgabe-Verzeichnis kopieren
            shutil.copy2(csv_path, os.path.join(output_path_dir, csv_file))

            # Bilder verarbeiten
            image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]
            total_images = len(image_files)

            print(f"Starting with folder {relative_path}, total images: {total_images}")

            processed_images = 0
            for filename in image_files:
                input_path = os.path.join(root, filename)
                img = cv2.imread(input_path)

                if img is None:
                    print(f"Warning: Image {input_path} could not be loaded and will be skipped.")
                    continue

                # Bild um 90 Grad im Uhrzeigersinn drehen
                img_rotated = rotate_image(img)

                # Gedrehtes Bild skalieren
                img_resized = cv2.resize(img_rotated, (xSize, ySize))

                edges = extract_and_enhance_edges(img_resized)
                final_img = apply_edge_overlay(img_resized, edges)

                output_path = os.path.join(output_path_dir, filename)
                cv2.imwrite(output_path, final_img)

                processed_images += 1
                print(f"Processed: {filename} - {processed_images}/{total_images} in current folder")

            processed_folders += 1
            print(f"Completed folder {processed_folders}/{total_folders}: {relative_path}")

    print("All images and CSV files have been processed and saved in the output folder.")

# Hauptausführung
if __name__ == "__main__":
    print("Starting image preprocessing for CNN with 90-degree rotation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    process_images_and_csv(input_dir, output_dir)

    print("Preprocessing completed successfully.")
