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
ImageProcessor.canny_low = 90
ImageProcessor.min_line_length = 5
ImageProcessor.max_line_gap = 2
ImageProcessor.hough_threshold = 10
# Pfad zum Verzeichnis mit den Bildern
dataset_path = 'datensatz/10.07/recordings_20240816_140117'
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

        cv2.imshow('Processed Imag2e', processed_image)

        # bild entzerren (perspektive korrigieren von fisheye mit 175°)
        
        
        lines = processor.detect_lines(processed_image)
        line_data = processor.extract_line_data(lines)
        line_data = processor.weight_lines(line_data)
        
        steering_line = processor.get_steering_line(img_resized, line_data)

        # zeichne das prozessierte kantenbild auf das bild
        if lines is not None:
            left_line, right_line = processor.get_right_and_left_line_mean(line_data)
            left_line_x = left_line.middle_x
            right_line_x = right_line.middle_x	

        diff_from_middle = (left_line_x + right_line_x) / 2 - xSize / 2
        print(f"Diff from middle: {diff_from_middle}")
        
        
        right_line.draw(img_resized, 255, 255, 255)
        left_line.draw(img_resized, 0, 255, 255)
        cv2.imshow('Birds Eye View', img_resized)

        if steering_line is not None:
            steering_line.draw(img_resized, 0, 255, 0)
            cv2.putText(img_resized, f"Steering Angle: {steering_line.calculate_angle():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        

        # Zeige das Bild mit Linien
        cv2.imshow('Processed Image', img_resized)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        

cv2.destroyAllWindows()