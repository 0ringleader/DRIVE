from imageProcessor import ImageProcessor
import cv2
import os

# Beispielverwendung:
def sort_pngs_dir_list(dir_list):
    def sort_pngs(file_name):
        number_part = file_name.split(".", 1)[0]
        return int(number_part)
    png_files = [file for file in dir_list if file.endswith('.png')]
    png_files.sort(key=sort_pngs)
    return png_files



# Größe der Bilder (hier als Beispiel gesetzt)
xSize = 256
ySize = 192

# Initialisiere den Bildprozessor
processor = ImageProcessor(xSize, ySize)

# Pfad zum Verzeichnis mit den Bildern
dataset_path = 'datensatz/mit_linie_in_der_mitte/recordings_20240808_143936'
print(os.listdir(dataset_path))
# Durchlaufe alle Bilder im Verzeichnis
for filename in sort_pngs_dir_list(os.listdir(dataset_path)):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(dataset_path, filename)
        raw_img = cv2.imread(img_path)
        img_rotated = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
        img_resized = cv2.resize(img_rotated, (xSize, ySize))

        # Bildverarbeitung und Lenkungsberechnung
        processed_image = processor.process_frame(img_resized)
        lines = processor.detect_lines(processed_image)
        line_data = processor.extract_line_data(lines)
        line_data = processor.weight_lines(line_data)
        for line in line_data:
            line.draw(img_resized)
        steering_line = processor.get_steering_line(img_resized, line_data)

        if steering_line is not None:
            steering_line.draw(img_resized)
            cv2.putText(img_resized, f"Steering Angle: {steering_line.calculate_angle():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Zeige das Bild mit Linien
        cv2.imshow('Processed Image', img_resized)
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()