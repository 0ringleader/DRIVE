import cv2
import numpy as np
import math

import cv2
import numpy as np
class line_object:
    def __init__(self, x1, y1, x2, y2, slope, angle, middle_x, middle_y, length, weight):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.slope = slope
        self.angle = angle
        self.middle_x = middle_x
        self.middle_y = middle_y
        self.length = length
        self.weight = weight
        self.left = False

    def __str__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}, Weight: {self.weight}, Left: {self.left}"

    def __repr__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}, Weight: {self.weight}, Left: {self.left}"

    def draw(self, image, r, g, b):

        
        # Berechne die um 180° gedrehten Koordinaten

        x1_rotated = self.middle_x + (self.middle_x - self.x1)
        y1_rotated = self.middle_y + (self.middle_y - self.y1)
        x2_rotated = self.middle_x + (self.middle_x - self.x2)
        y2_rotated = self.middle_y + (self.middle_y - self.y2)

        # Zeichne den gedrehten Pfeil
        cv2.arrowedLine(image, (int(x1_rotated), int(y1_rotated)), (int(x2_rotated), int(y2_rotated)), (r,g,b), 2, tipLength=0.05)
        
        # Zeichne den Mittelpunkt
        cv2.circle(image, (int(self.middle_x), int(self.middle_y)), 5, (0, 0, 255), -1)
        
        # Zeige das Bild

    def calculate_angle(self):
        x1 = self.x1
        y1 = self.y1
        x2 = self.x2
        y2 = self.y2
        
        delta_x = x2 - x1
        delta_y = y2 - y1

        theta_rad = np.arctan2(delta_x, delta_y)
        theta_deg = np.degrees(theta_rad)

        angle = theta_deg
        return angle

        

    #packt alle daten in ein eindiemensionales array
    def flatten(self):
        return [self.x1, self.y1, self.x2, self.y2, self.slope, self.angle, self.middle_x, self.middle_y, self.length, self.weight]

class ImageProcessor:
    def __init__(self, x_size, y_size, gaussian_blur=5, canny_low_threshold=60, canny_high_threshold=200, hough_rho=1, hough_theta=np.pi/90, hough_threshold=35, hough_min_line_length=10, hough_max_line_gap=10):
        self.x_size = x_size
        self.y_size = y_size
        self.gaussian_blur = gaussian_blur
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        
    
    def process_frame(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur, self.gaussian_blur), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(blurred)

        edges = cv2.Canny(cl1, self.canny_low_threshold, self.canny_high_threshold)

        height, width = edges.shape
        mask = np.zeros_like(edges)

        polygon = np.array([
            [(0, height * 1/2),
            (width, height * 1/2),
            (width, height),
            (0, height)]
        ], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges

    def detect_lines(self, image):
        lines = cv2.HoughLinesP(image, rho=self.hough_rho, theta=self.hough_theta, threshold=self.hough_threshold, minLineLength=self.hough_min_line_length, maxLineGap=self.hough_max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return lines


    def extract_line_data(self, lines):
        line_data = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                if x1 == x2:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                middle_x = (x1 + x2) / 2
                middle_y = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                line_obj = line_object(x1, y1, x2, y2, slope, angle, middle_x, middle_y, length, 0)
                line_obj.left = slope < 0

                line_data.append(line_obj)
        return line_data

    def weight_lines(self, lines):
        if len(lines) < 3:
            return lines

        max_length = max(line.length for line in lines)
        max_y = self.y_size
        max_x = self.x_size

        for line in lines:
            length_weight = line.length / max_length if max_length != 0 else 0
            y_weight = 1 - (line.middle_y / max_y)
            x_weight = 1 - abs(line.middle_x - max_x / 2) / (max_x / 2)

            line.weight = length_weight * y_weight * x_weight

        left_weight = sum(line.weight for line in lines if line.left)
        right_weight = sum(line.weight for line in lines if not line.left)

        for line in lines:
            if line.left:
                line.weight = line.weight / left_weight * 0.5 if left_weight != 0 else 0
            else:
                line.weight = line.weight / right_weight * 0.5 if right_weight != 0 else 0

        return lines

    def get_steering_line(self, image, line_data):
        line_object_sum = line_object(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        total_weight = sum(line.weight for line in line_data)

        if total_weight == 0:
            print(line_data)
            print("Fehler: Gesamtgewicht der Linien ist 0. Überprüfe die Gewichtungsfunktion.")
            return None

        for line in line_data:
            line_object_sum.x1 += line.x1 * line.weight
            line_object_sum.y1 += line.y1 * line.weight
            line_object_sum.x2 += line.x2 * line.weight
            line_object_sum.y2 += line.y2 * line.weight
            line_object_sum.middle_x += line.middle_x * line.weight
            line_object_sum.middle_y += line.middle_y * line.weight
            line_object_sum.length += line.length * line.weight
            line_object_sum.weight += line.weight

        line_object_sum.x1 = int(line_object_sum.x1 / line_object_sum.weight)
        line_object_sum.y1 = int(line_object_sum.y1 / line_object_sum.weight)
        line_object_sum.x2 = int(line_object_sum.x2 / line_object_sum.weight)
        line_object_sum.y2 = int(line_object_sum.y2 / line_object_sum.weight)
        line_object_sum.middle_x = int(line_object_sum.middle_x / line_object_sum.weight)
        line_object_sum.middle_y = int(line_object_sum.middle_y / line_object_sum.weight)

        if line_object_sum.x2 - line_object_sum.x1 != 0:
            line_object_sum.slope = (line_object_sum.y2 - line_object_sum.y1) / (line_object_sum.x2 - line_object_sum.x1)
        else:
            line_object_sum.slope = float('inf')

        self.angle = line_object_sum.calculate_angle()

        return line_object_sum
    

#gibt die gemittelte rechte und linke linie zurück aus einem array von linien
    def get_right_and_left_line_mean(self, lines):
        left_lines = []
        right_lines = []
        for line in lines:
            if line.left:
                left_lines.append(line)
            else:
                right_lines.append(line)
        left_line = self.get_mean_line(left_lines)
        right_line = self.get_mean_line(right_lines)

        if left_line is None and right_line is None:
            return None, None

        if left_line is None: # Wenn keine Linie gefunden wurde, wird die linke Linie ganz links gesetzt und leicht nach links geneigt
            left_line = right_line

        if right_line is None: # Wenn keine Linie gefunden wurde, wird die rechte Linie ganz rechts gesetzt und leicht nach rechts geneigt
            right_line = left_line
        

        return left_line, right_line
    
    def get_mean_line(self, lines):
        if len(lines) == 0:
            return None

        x1 = sum(line.x1 for line in lines) / len(lines)
        y1 = sum(line.y1 for line in lines) / len(lines)
        x2 = sum(line.x2 for line in lines) / len(lines)
        y2 = sum(line.y2 for line in lines) / len(lines)
        slope = sum(line.slope for line in lines) / len(lines)
        angle = sum(line.angle for line in lines) / len(lines)
        middle_x = sum(line.middle_x for line in lines) / len(lines)
        middle_y = sum(line.middle_y for line in lines) / len(lines)
        length = sum(line.length for line in lines) / len(lines)
        weight = sum(line.weight for line in lines) / len(lines)



        return line_object(x1, y1, x2, y2, slope, angle, middle_x, middle_y, length, weight)

# # # Beispielverwendung:
# # xSize = 256
# # ySize = 192
# # processor = ImageProcessor(xSize, ySize)

# # img = cv2.imread('path_to_image')
# # processed_image = processor.process_frame(img)
# # lines = processor.detect_lines(processed_image)
# # line_data = processor.extract_line_data(lines)
# # line_data = processor.weight_lines(line_data)
# # steering_line = processor.get_steering_line(img, line_data)

# # if steering_line is not None:
# #     steering_line.draw(img)
# #     cv2.putText(img, f"Steering Angle: {steering_line.angle:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # cv2.imshow('lines', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()