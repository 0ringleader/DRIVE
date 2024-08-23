#precalc.py edited 2308 @9:50AM by Sven

import cv2
import numpy as np
import os
import shutil
import logging
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import collections

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration variables
config = ConfigParser()
config.read('config.ini')

# Helper functions for conversion and type checking
def get_bool(section, var_name, default=None):
    try:
        return config.getboolean(section, var_name, fallback=default)
    except ValueError:
        logging.warning(
            f"The value for {var_name} in {section} cannot be converted to boolean. Using default value {default}.")
        return default

def get_float(section, var_name, default=None):
    try:
        return config.getfloat(section, var_name, fallback=default)
    except ValueError:
        logging.warning(
            f"The value for {var_name} in {section} cannot be converted to float. Using default value {default}.")
        return default

def get_int(section, var_name, default=None):
    try:
        return config.getint(section, var_name, fallback=default)
    except ValueError:
        logging.warning(
            f"The value for {var_name} in {section} cannot be converted to int. Using default value {default}.")
        return default

def get_tuple(section, var_name, default=None):
    value = config.get(section, var_name, fallback=default)
    if isinstance(value, tuple):
        return value
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        logging.warning(
            f"The value for {var_name} in {section} cannot be converted to tuple of int. Using default value {default}.")
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
rotate = get_bool('precalc', 'rotate', True)
apply_filter = get_bool('precalc', 'apply_filter', True)
apply_canny = get_bool('precalc', 'apply_canny', True)
draw_center_line = get_bool('precalc', 'drawcenterline', True)
centerline_sensitivity = get_float('precalc', 'centerline_sensitivity', 1.0)

# Input and output directories
input_dir = "PreCalcIn"
output_dir = "PreCalcOut"
os.makedirs(output_dir, exist_ok=True)

# Function to adjust contrast and brightness
def adjust_contrast_brightness(image_gray, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image_gray, alpha=alpha, beta=beta)

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
    binary_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            adaptive_thresh_block_size, adaptive_thresh_C)
    # Combine the two results by taking the logical AND
    binary_combined = cv2.bitwise_and(binary_global, binary_adaptive)
    return binary_combined

# Function to detect edges using Canny
def apply_canny_edge_detection(image_binary):
    return cv2.Canny(image_binary, 100, 200)

# Function to crop the image (if needed to focus on specific region)
def crop_image(image):
    height, width = image.shape[:2]
    cropped = image[height // 2:, :]
    return cropped

# Function to find lane lines using Hough Transform
def find_lane_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    left_lines = []
    right_lines = []
    img_center = image.shape[1] // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0  # avoid divide by zero
            if abs(slope) < 0.5:  # filter near-horizontal lines
                continue
            if slope < 0 and x1 < img_center and x2 < img_center:  # left lane
                left_lines.append(line)
            elif slope > 0 and x1 > img_center and x2 > img_center:  # right lane
                right_lines.append(line)
    return left_lines, right_lines

# Function to average and extrapolate lane lines
def average_lane_lines(image, lines):
    if len(lines) == 0:
        return None
    slope_intercepts = [(np.polyfit((line[0][0], line[0][2]), (line[0][1], line[0][3]), 1)) for line in lines]
    slope_intercept_avg = np.average(slope_intercepts, axis=0)
    slope, intercept = slope_intercept_avg

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [(x1, y1, x2, y2)]

# Function to draw lines on an image
def draw_lines(image, lines, color=(255, 0, 0), thickness=5):
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = map(int, line) # Ensure coordinates are integers
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Function to process the image through a customizable pipeline
def process_image_pipeline(image, steps):
    for step in steps:
        image = step(image)
    return image

# Keep track of previous lane lines for fallback
previous_left_line = None
previous_right_line = None

# Deque to store the last few center line calculations for smoothing
center_line_queue = collections.deque(maxlen=5)

def calculate_moving_average(values):
    return np.mean(values, axis=0)

# Updated function to process a single image
def process_single_image(input_path, output_path_dir, filename):
    global previous_left_line, previous_right_line, center_line_queue

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Image {input_path} could not be loaded")
    except ValueError as e:
        logging.error(e)
        return

    # Resize image
    img_resized = cv2.resize(img, (xSize, ySize))

    # Rotate image if the rotate parameter is True
    if rotate:
        img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Define the processing pipeline
    steps = [adjust_contrast_brightness, apply_clahe]

    if apply_filter:
        steps.append(reduce_noise)

    steps.append(binarize_image)

    if apply_canny:
        steps.append(apply_canny_edge_detection)

    # Process image through the pipeline
    img_processed = process_image_pipeline(img_gray, steps)

    # Optionally crop the image to focus on the relevant region (e.g., road)
    img_cropped = crop_image(img_processed)

    # Draw the lane and center lines on a copy of the original resized image
    lane_lines_img = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)  # Convert to BGR to draw the red line

    # Find lane lines
    left_lines, right_lines = find_lane_lines(img_cropped)

    # Average and extrapolate lane lines, fallback to previous if necessary
    left_line = average_lane_lines(img_cropped, left_lines) if left_lines else previous_left_line
    right_line = average_lane_lines(img_cropped, right_lines) if right_lines else previous_right_line

    # Update previous lines
    if left_line:
        previous_left_line = left_line
    if right_line:
        previous_right_line = right_line

    # Draw centerline if draw_center_line is True
    if draw_center_line and left_line and right_line:
        center_line = [
            (
                int((left_line[0][0] + centerline_sensitivity * right_line[0][0]) / (1 + centerline_sensitivity)),
                int((left_line[0][1] + centerline_sensitivity * right_line[0][1]) / (1 + centerline_sensitivity)),
                int((left_line[0][2] + centerline_sensitivity * right_line[0][2]) / (1 + centerline_sensitivity)),
                int((left_line[0][3] + centerline_sensitivity * right_line[0][3]) / (1 + centerline_sensitivity))
            )
        ]

        # Update the deque with the new center line and calculate the smoothed center line
        center_line_queue.append(center_line)
        smoothed_center_line = calculate_moving_average(center_line_queue)

        # Draw the smoothed center line
        draw_lines(lane_lines_img, smoothed_center_line, color=(0, 0, 255), thickness=2)  # Red for center line

    # Convert back to grayscale, if needed
    img_final = cv2.cvtColor(lane_lines_img, cv2.COLOR_BGR2GRAY)

    # Save the final image
    output_path = os.path.join(output_path_dir, filename)
    cv2.imwrite(output_path, img_final)

# Main processing function for images and CSV files
def process_images_and_csv(input_dir, output_dir):
    total_folders = sum(1 for _, dirnames, _ in os.walk(input_dir) if not dirnames)
    processed_folders = 0

    for root, _, files in tqdm(os.walk(input_dir), total=total_folders, desc="Processing Folders"):
        csv_file = next((f for f in files if f.endswith('.csv')), None)
        image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]

        # Check if the folder is complete, i.e., it contains both images and a CSV file
        if not (csv_file and image_files):
            logging.warning(f"Folder {os.path.relpath(root, input_dir)} is incomplete and will be skipped.")
            continue

        csv_path = os.path.join(root, csv_file)
        relative_path = os.path.relpath(root, input_dir)
        output_path_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_path_dir, exist_ok=True)

        # Copy CSV file to output directory
        shutil.copy2(csv_path, os.path.join(output_path_dir, csv_file))

        total_images = len(image_files)
        logging.info(f"Starting with folder {relative_path}, total images: {total_images}")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for filename in image_files:
                input_path = os.path.join(root, filename)
                futures.append(executor.submit(process_single_image, input_path, output_path_dir, filename))

            for i, future in enumerate(as_completed(futures)):
                future.result()  # This will raise any exceptions that occurred during processing
                logging.info(f"Processed: {i + 1}/{total_images} in current folder")

        processed_folders += 1
        logging.info(f"Completed folder {processed_folders}/{total_folders}: {relative_path}")

    logging.info("All images and CSV files have been processed and saved in the output folder.")

# Main execution
if __name__ == "__main__":
    logging.info(
        "Starting enhanced image preprocessing for CNN with grayscale conversion, optional noise reduction, and binarization...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    process_images_and_csv(input_dir, output_dir)

    logging.info("Preprocessing completed successfully.")
