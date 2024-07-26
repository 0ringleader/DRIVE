import cv2
import numpy as np
import os
from scipy.interpolate import UnivariateSpline

# Parameter
xSize = 256
ySize = 192

# Verzeichnis für die Eingabebilder
input_dir = "PreCalcIn"

# Verzeichnis für die Ausgabe verarbeiteter Bilder
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

def deblur_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def extract_curves(image):
    # In Graustufen konvertieren
    image = deblur_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Entweder Schärfen oder Deblurring anwenden
    sharpened_img = sharpen_image(gray)  # Versuchen Sie stattdessen deblur_image
    # Rauschunterdrückung
    blur = cv2.GaussianBlur(sharpened_img, (9, 9), 0)  # Größeren Kernel verwenden

    # Kantenerkennung
    edges = cv2.Canny(blur, 0, 100)  # Schwellenwerte anpassen

    # Finden der Konturen
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def fit_bezier_curve(contour):
    points = contour.reshape(-1, 2)
    
    if len(points) < 4:
        return None

    x = points[:, 0]
    y = points[:, 1]

    spline_x = UnivariateSpline(np.arange(len(x)), x, k=3, s=0)
    spline_y = UnivariateSpline(np.arange(len(y)), y, k=3, s=0)

    t = np.linspace(0, len(x) - 1, num=1000)
    bezier_x = spline_x(t)
    bezier_y = spline_y(t)

    return bezier_x, bezier_y

def create_bezier_mask(image, bezier_curves):
    mask = np.zeros_like(image[:, :, 0])

    for bezier_x, bezier_y in bezier_curves:
        for i in range(len(bezier_x) - 1):
            cv2.line(mask, (int(bezier_x[i]), int(bezier_y[i])), (int(bezier_x[i+1]), int(bezier_y[i+1])), 255, 2)
    
    return mask

# Durchlaufe alle Dateien im Eingabeverzeichnis
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Fehler beim Laden des Bildes {filename}.")
            continue

        img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img_resized = cv2.resize(img_rotated, (xSize, ySize))

        contours = extract_curves(img_resized)

        bezier_curves = []
        for contour in contours:
            bezier_curve = fit_bezier_curve(contour)
            if bezier_curve:
                bezier_curves.append(bezier_curve)

        bezier_mask = create_bezier_mask(img_resized, bezier_curves)

        masked_img = cv2.bitwise_and(img_resized, img_resized, mask=bezier_mask)

        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, masked_img)

        cv2.imshow('Masked Image', masked_img)
        cv2.waitKey(5)

        # Debugging: Zeige die Kantenerkennung und ROI an
        cv2.imshow('Edges', cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY), (9, 9), 0), 30, 100))
        cv2.imshow('Contour Mask', bezier_mask)

cv2.destroyAllWindows()

print("Bilder wurden erfolgreich verarbeitet und im Output-Ordner gespeichert.")
