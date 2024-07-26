import cv2
import numpy as np
import os
from scipy.special import comb

# Parameters
xSize = 256
ySize = 192

# Verzeichnis für die Eingabebilder
input_dir = "PreCalcIn"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut2"
os.makedirs(output_dir, exist_ok=True)

def extract_contrast_regions(image):
    # In Graustufen konvertieren
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Verwende adaptive Schwellenwertbestimmung
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Die Konturen der Linie finden
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtere die Konturen basierend auf Form-Parametern
    filtered_contours = []
    for contour in contours:
        # Berechne die Begrenzungsbox der Kontur
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtere basierend auf dem Verhältnis von Breite zu Höhe und der Größe der Kontur
        aspect_ratio = float(w) / h
        if 0.5 < aspect_ratio < 5 and cv2.contourArea(contour) > 50:  # Angepasste Werte
            filtered_contours.append(contour)

    # Eine leere Maske erstellen, um die Linie zu zeichnen
    mask = np.zeros_like(gray)

    # Konturen auf der Maske zeichnen
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Anzahl der Linien bestimmen
    num_lines = len(filtered_contours)

    return mask, num_lines, filtered_contours

def bezier_curve(points, num_points=100):
    """
    Berechnet eine Bézier-Kurve für eine gegebene Liste von Punkten.
    """
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    
    # Berechnung der Bézier-Kurve
    for i in range(n + 1):
        binom = comb(n, i)
        curve += binom * ((1 - t) ** (n - i))[:, np.newaxis] * (t ** i)[:, np.newaxis] * np.array(points[i])
    
    return curve

def draw_bezier_contours(image, contours):
    """
    Zeichnet Bézier-Kurven auf den Konturen des Bildes.
    """
    for contour in contours:
        # Die Kontur in eine Liste von Punkten umwandeln
        contour = contour.reshape(-1, 2)
        if len(contour) > 2:
            # Reduziere die Punktanzahl durch den Douglas-Peucker-Algorithmus
            contour_reduced = cv2.approxPolyDP(contour, epsilon=1, closed=True)  # Setze epsilon für die Reduzierung
            bezier_points = bezier_curve(contour_reduced, num_points=50)
            bezier_points = np.int32(bezier_points)
            
            # Zeichne die Bézier-Kurve
            for i in range(len(bezier_points) - 1):
                cv2.line(image, tuple(bezier_points[i]), tuple(bezier_points[i + 1]), (0, 0, 255), 2)


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
        mask, num_lines, contours = extract_contrast_regions(img_resized)

        # Maskierte Bild anzeigen
        masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Zeichne Bézier-Kurven auf den Konturen
        draw_bezier_contours(np.ones(img_resized.shape), contours)

        # Pfad zum Ausgabebild
        output_path = os.path.join(output_dir, filename)
        
        # Bild anzeigen
        cv2.imshow('masked', mask)
        cv2.imshow('contours', img_resized)

        # Bild speichern
        cv2.imwrite(output_path, mask)
        
        print(f"Bild {filename}: {num_lines} Linien erkannt.")

        # Wenn 'q' gedrückt wird, wird das Programm beendet
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
