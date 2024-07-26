import cv2
import numpy as np
import os

# Parameters
xSize = 128
ySize = 96

# Verzeichnis für die Eingabebilder
input_dir = "PreCalcIn"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

def extract_contrast_regions(image):
    # In Graustufen konvertieren
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Schwellenwert kann angepasst werden
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Die Konturen der Linie finden
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtere die Konturen basierend auf Form-Parametern
    # filtered_contours = []
    filtered_contours = contours


    # Eine leere Maske erstellen, um die Linie zu zeichnen
    mask = np.zeros_like(gray)

    # Konturen auf der Maske zeichnen
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Anzahl der Linien bestimmen
    num_lines = len(filtered_contours)

    return mask, num_lines

# Durchlaufe alle Dateien im Eingabeverzeichnis
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Pfad zum Eingabebild
        input_path = os.path.join(input_dir, filename)

        # Bild öffnen
        img = cv2.imread(input_path)
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Bild auf 256x192 skalieren
        img_resized = cv2.resize(img_rotated, (xSize, ySize))

        # Kontrastregionen im Bild extrahieren
        mask, num_lines = extract_contrast_regions(img_resized)

        # Maske anwenden, um die ausgewählten Bereiche im Bild hervorzuheben
        masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Pfad zum Ausgabebild
        output_path = os.path.join(output_dir, filename)
        
        # Bild anzeigen
        cv2.imshow('masked', masked_img)
        
        # Bild speichern
        cv2.imwrite(output_path, masked_img)
        
        print(f"Bild {filename}: {num_lines} Linien erkannt.")

        # Wenn 'q' gedrückt wird, wird das Programm beendet
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
