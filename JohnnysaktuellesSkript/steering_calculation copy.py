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
processor.canny_low = 60
processor.min_line_length = 5
processor.max_line_gap = 2
# Pfad zum Verzeichnis mit den Bildern
dataset_path = 'C:/_Code/TH/DRIVE/datensatz/10.07/recordings_20240816_130337'
print(os.listdir(dataset_path))
# Durchlaufe alle Bilder im Verzeichnis
for filename in sort_pngs_dir_list(os.listdir(dataset_path)):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(dataset_path, filename)
        raw_img = cv2.imread(img_path)
        
        
        img_rotated = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
        img_resized = cv2.resize(img_rotated, (xSize, ySize))
        processor.preprocess_frame(img_resized)
        cv2.imshow('preprocessed', processor.image)
        processor.detect_lines()
        processor.sort_lines()
        cv2.imshow('processed', processor.raw_image) 
        


 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        

cv2.destroyAllWindows()