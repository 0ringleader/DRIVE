import cv2
import numpy as np
import os

# Parameters
xSize = 256
ySize = 192

# Verzeichnis für die Eingabebilder
input_dir = "PreCalcIn"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

def extract_contrast_regions(image):
    # Konvertierung in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Anwendung des Gaussian Blur, um Rauschen zu reduzieren
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny-Kantenerkennung
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Morphologie-Operationen (Dilatation gefolgt von Erosion) für bessere Kantenverbindungen
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Finde Konturen in der binären Mask
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask

def normalize_image(image):
    # Normalisiere das Bild auf den Bereich [0, 1]
    return image / 255.0

# Durchlaufe alle Dateien im Eingabeverzeichnis
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Pfad zum Eingabebild
        input_path = os.path.join(input_dir, filename)

        # Bild öffnen
        img = cv2.imread(input_path)

        # Bild auf 256x192 skalieren
        img_resized = cv2.resize(img, (xSize, ySize))

        # Kontrastregionen im Bild extrahieren
        mask = extract_contrast_regions(img_resized)

        # Maske anwenden, um die ausgewählten Bereiche im Bild hervorzuheben
        masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Bild auf den Bereich [0, 1] normalisieren
        normalized_img = normalize_image(masked_img)

        # Pfad zum Ausgabebild
        output_path = os.path.join(output_dir, filename)

        # Bild speichern
        cv2.imwrite(output_path, (normalized_img * 255).astype(np.uint8))

print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")