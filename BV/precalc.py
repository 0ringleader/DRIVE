#precalc.py edited 1408 @9:25AM by Sven

import cv2
import numpy as np
import os
import shutil
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load configuration variables
config = ConfigParser()
config.read('config.ini')


# Helper functions for conversion and type checking
def get_bool(section, var_name, default=None):
    try:
        return config.getboolean(section, var_name, fallback=default)
    except ValueError:
        print(
            f"Warning: The value for {var_name} in {section} cannot be converted to boolean. Using default value {default}.")
        return default


def get_float(section, var_name, default=None):
    try:
        return config.getfloat(section, var_name, fallback=default)
    except ValueError:
        print(
            f"Warning: The value for {var_name} in {section} cannot be converted to float. Using default value {default}.")
        return default


def get_int(section, var_name, default=None):
    try:
        return config.getint(section, var_name, fallback=default)
    except ValueError:
        print(
            f"Warning: The value for {var_name} in {section} cannot be converted to int. Using default value {default}.")
        return default


def get_tuple(section, var_name, default=None):
    value = config.get(section, var_name, fallback=default)
    if isinstance(value, tuple):
        return value
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        print(
            f"Warning: The value for {var_name} in {section} cannot be converted to tuple of int. Using default value {default}.")
        return default


# Get variables from the configuration file and convert
xSize = get_int('precalc', 'xsize', 256)
ySize = get_int('precalc', 'ysize', 192)
clahe_clipLimit = get_float('precalc', 'clahe_cliplimit', 3.0)
clahe_tileGridSize = get_tuple('precalc', 'clahe_tilegridsize', (8, 8))
gaussian_blur_ksize = get_int('precalc', 'gaussian_blur_ksize', 5)
denoising_h = get_float('precalc', 'denoising_h', 30)
denoising_template_window_size = get_int('precalc', 'denoising_template_window_size', 7)
denoising_search_window_size = get_int('precalc', 'denoising_search_window_size', 21)
adaptive_thresh_block_size = get_int('precalc', 'adaptive_thresh_block_size', 11)
adaptive_thresh_C = get_float('precalc', 'adaptive_thresh_C', 2)

# New variables for rotation and filtering
rotate = get_bool('precalc', 'rotate', True)
apply_filter = get_bool('precalc', 'apply_filter', False)

# Input and output directories
input_dir = "PreCalcIn"
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)


# Function to apply CLAHE
def apply_clahe(image_gray):
    clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
    return clahe.apply(image_gray)


# Function to reduce noise
def reduce_noise(image_gray):
    # Apply Gaussian Blur
    ksize = (gaussian_blur_ksize, gaussian_blur_ksize)
    blurred = cv2.GaussianBlur(image_gray, ksize, 0)

    # Apply Denoising
    denoised = cv2.fastNlMeansDenoising(blurred, None, denoising_h, denoising_template_window_size,
                                        denoising_search_window_size)

    return denoised


# Function to binarize the image
def binarize_image(image_gray):
    # Apply global thresholding
    _, binary_global = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, adaptive_thresh_block_size, adaptive_thresh_C)

    # Combine the two results by taking the logical AND
    binary_combined = cv2.bitwise_and(binary_global, binary_adaptive)

    return binary_combined


# Updated function to process a single image
def process_single_image(input_path, output_path_dir, filename):
    img = cv2.imread(input_path)

    if img is None:
        print(f"Warning: Image {input_path} could not be loaded and will be skipped.")
        return

    # Resize image
    img_resized = cv2.resize(img, (xSize, ySize))

    # Rotate image if the rotate parameter is True
    if rotate:
        img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    img_clahe = apply_clahe(img_gray)

    # Reduce noise if apply_filter is True
    if apply_filter:
        img_clahe = reduce_noise(img_clahe)

    # Binarize image
    img_binary = binarize_image(img_clahe)

    # Save the final image
    output_path = os.path.join(output_path_dir, filename)
    cv2.imwrite(output_path, img_binary)


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
    print(
        "Starting enhanced image preprocessing for CNN with grayscale conversion, optional noise reduction, and binarization...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    process_images_and_csv(input_dir, output_dir)

    print("Preprocessing completed successfully.")
