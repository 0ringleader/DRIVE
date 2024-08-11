#precalc.py edited 1108 @11:50PM by Sven

import cv2
import numpy as np
import os
import csv
import shutil
from setVariables import SetVariables, replySetVariables
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load configuration variables
config = SetVariables('config.ini')
variables = config.get_variables('precalc.py')


# Helper functions for conversion and type checking
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


# Get variables from the configuration file and convert
xSize = get_int('xSize', 256)
ySize = get_int('ySize', 192)
edge_strength = get_float('edge_strength', 1.0)
noise_h = get_float('noise_h', 10)
noise_hColor = get_float('noise_hColor', 10)
noise_templateWindowSize = get_int('noise_templateWindowSize', 7)
noise_searchWindowSize = get_int('noise_searchWindowSize', 21)
canny_threshold1 = get_float('canny_threshold1', 50)
canny_threshold2 = get_float('canny_threshold2', 150)
clahe_clipLimit = get_float('clahe_clipLimit', 3.0)
clahe_tileGridSize = get_tuple('clahe_tileGridSize', (8, 8))

# Input and output directories
input_dir = "PreCalcIn"
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

# Output response to set variables
replySetVariables('precalc.py')


# Image processing functions
def reduce_artifacts(edges):
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    return edges


def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, noise_h, noise_hColor, noise_templateWindowSize,
                                           noise_searchWindowSize)


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
    edge_color = (0, 255, 0)  # Green color for edges
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color_edges[np.where((color_edges == [255, 255, 255]).all(axis=2))] = edge_color
    return cv2.addWeighted(image, 1, color_edges, edge_strength, 0)


def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


# New function for data augmentation
def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Vary brightness
    brightness = np.random.uniform(0.8, 1.2)
    augmented_images.append(cv2.convertScaleAbs(image, alpha=brightness, beta=0))

    # Vary contrast
    contrast = np.random.uniform(0.8, 1.2)
    augmented_images.append(cv2.convertScaleAbs(image, alpha=contrast, beta=0))

    # Random rotation (small angles)
    angle = np.random.uniform(-10, 10)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))

    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))

    return augmented_images


# New function for adaptive parameter adjustment
def adaptive_parameters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Adjust parameters based on image properties
    adjusted_canny_threshold1 = max(10, min(100, canny_threshold1 * (mean_brightness / 128)))
    adjusted_canny_threshold2 = max(50, min(200, canny_threshold2 * (mean_brightness / 128)))
    adjusted_clahe_clipLimit = max(1.0, min(5.0, clahe_clipLimit * (std_brightness / 64)))

    return adjusted_canny_threshold1, adjusted_canny_threshold2, adjusted_clahe_clipLimit


# Updated function to process a single image
def process_single_image(input_path, output_path_dir, filename):
    img = cv2.imread(input_path)

    if img is None:
        print(f"Warning: Image {input_path} could not be loaded and will be skipped.")
        return

    # Adaptive parameter adjustment
    adj_canny_threshold1, adj_canny_threshold2, adj_clahe_clipLimit = adaptive_parameters(img)

    augmented_images = augment_image(img)

    for i, aug_img in enumerate(augmented_images):
        # Rotate image 90 degrees clockwise
        aug_img_rotated = rotate_image(aug_img)

        # Resize rotated image
        aug_img_resized = cv2.resize(aug_img_rotated, (xSize, ySize))

        # Apply adaptive parameters
        clahe = cv2.createCLAHE(clipLimit=adj_clahe_clipLimit, tileGridSize=clahe_tileGridSize)
        l, a, b = cv2.split(cv2.cvtColor(aug_img_resized, cv2.COLOR_BGR2LAB))
        cl = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        edges = cv2.Canny(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY),
                          threshold1=adj_canny_threshold1,
                          threshold2=adj_canny_threshold2)
        edges = reduce_artifacts(edges)

        final_img = apply_edge_overlay(enhanced_img, edges)

        output_filename = f"{os.path.splitext(filename)[0]}_aug{i}{os.path.splitext(filename)[1]}"
        output_path = os.path.join(output_path_dir, output_filename)
        cv2.imwrite(output_path, final_img)


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

            # Copy CSV file to output directory
            shutil.copy2(csv_path, os.path.join(output_path_dir, csv_file))

            # Process images
            image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]
            total_images = len(image_files)

            print(f"Starting with folder {relative_path}, total images: {total_images}")

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = []
                for filename in image_files:
                    input_path = os.path.join(root, filename)
                    futures.append(executor.submit(process_single_image, input_path, output_path_dir, filename))

                for i, future in enumerate(as_completed(futures)):
                    future.result()  # This will raise any exceptions that occurred during processing
                    print(f"Processed: {i + 1}/{total_images} in current folder")

            processed_folders += 1
            print(f"Completed folder {processed_folders}/{total_folders}: {relative_path}")

    print("All images and CSV files have been processed and saved in the output folder.")


# Main execution
if __name__ == "__main__":
    print("Starting enhanced image preprocessing for CNN with 90-degree rotation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    process_images_and_csv(input_dir, output_dir)

    print("Preprocessing completed successfully.")
