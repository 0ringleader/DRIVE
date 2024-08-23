import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

import cv2
import numpy as np
class line_object:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        #calculate slope
        if x2 - x1 != 0:
            self.slope = (y2 - y1) / (x2 - x1)
        else:
            self.slope = float('inf')
        #calculate angle
        self.angle = self.calculate_angle()
        #calculate middle
        self.middle_x = (x1 + x2) / 2
        self.middle_y = (y1 + y2) / 2
        #calculate length
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        #weight
        self.weight = 1
        

    def __str__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}"

    def __repr__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Angle: {self.angle}, Middle: ({self.middle_x}, {self.middle_y}), Length: {self.length}, Weight: {self.weight}, Left: {self.left}"
    def draw(self, image = None, color = (0, 255, 0)):

        
        # Berechne die um 180° gedrehten Koordinaten

        x1_rotated = self.middle_x + (self.middle_x - self.x1)
        y1_rotated = self.middle_y + (self.middle_y - self.y1)
        x2_rotated = self.middle_x + (self.middle_x - self.x2)
        y2_rotated = self.middle_y + (self.middle_y - self.y2)

        # Zeichne den gedrehten Pfeil
        cv2.arrowedLine(image, (int(x1_rotated), int(y1_rotated)), (int(x2_rotated), int(y2_rotated)), color, 2, tipLength=0.05)
        
        # Zeichne den Mittelpunkt
        cv2.circle(image, (int(self.middle_x), int(self.middle_y)), 5, color, -1)
        
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




class ImageProcessor:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.image = None
        self.raw_image = None
        self.hough_lines = None
        self.lines = None
        self.steering_val = 0
        self.extraSteering = 50

    def preprocess_frame(self, image):
        self.raw_image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(blurred)

        edges = cv2.Canny(cl1, 100, 200)

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
        self.image = cropped_edges


    def detect_lines(self):
        self.lines = []
        self.hough_lines = cv2.HoughLinesP(self.image, rho=1, theta=np.pi/90, threshold=25, minLineLength=40, maxLineGap=10)
        if self.hough_lines is not None:
            for line in self.hough_lines:
                x1, y1, x2, y2 = line[0]
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                if x1 == x2:
                    continue

                line_obj = line_object(x1, y1, x2, y2)

                self.lines.append(line_obj)


    def sort_lines(self):
        self.lines.sort(key=lambda x: x.middle_y)
        left_lines = []
        right_lines = []
        lowest_left_line = None
        lowest_right_line = None
        middle_x = None
        steering_val = None


        for line in self.lines:
            if line.slope < 0:
                left_lines.append(line)
            elif line.slope > 0:
                right_lines.append(line)
        
        if len(left_lines) > 0 and len(right_lines) > 0:
            left_median_x = np.median([line.middle_x for line in left_lines])
            right_median_x = np.median([line.middle_x for line in right_lines])
            #sort out all left lines that are more to the right than the median of the right lines
            left_lines = [line for line in left_lines if line.middle_x < right_median_x]
            #sort out all right lines that are more to the left than the median of the left lines
            right_lines = [line for line in right_lines if line.middle_x > left_median_x]
            #sort out left lines that are on the right half of the image
            left_lines = [line for line in left_lines if line.middle_x < self.x_size / 2]
            #sort out right lines that are on the left half of the image
            right_lines = [line for line in right_lines if line.middle_x > self.x_size / 2]



        if len(left_lines) > 0 and len(right_lines) > 0:

            left_lines.sort(key=lambda x: x.middle_y)
            right_lines.sort(key=lambda x: x.middle_y)
            lowest_left_line = left_lines[-1]
            lowest_right_line = right_lines[-1]
            lowest_left_line.draw(self.raw_image, (0, 255, 0))
            lowest_right_line.draw(self.raw_image, (0, 0, 255))
            #berechne den x-Wert vom schnittpunkt der linken linie mit dem unteren Bildrand
            x_left = lowest_left_line.middle_x - (lowest_left_line.middle_y - self.y_size) / lowest_left_line.slope
            #berechne den x-Wert vom schnittpunkt der rechten linie mit dem unteren Bildrand
            x_right = lowest_right_line.middle_x - (lowest_right_line.middle_y - self.y_size) / lowest_right_line.slope
            #berechne den abstand zur bildmitte der schnittpunkte
            middle_x = (x_left + x_right) / 2

            self.steering_val = (middle_x - self.x_size / 2)
            cv2.circle(self.raw_image, (int(middle_x), 50), 5, (255, 0, 0), -1)
        else:
            #nur die linke Linie ist vorhanden
            if len(left_lines) > 0:
                left_lines.sort(key=lambda x: x.middle_y)
                lowest_left_line = left_lines[-1]
                lowest_left_line.left = True
                lowest_left_line.draw(self.raw_image, (0, 255, 0)) #draw the line in green
                #diese formel kann man noch optimieren
                middle_x = (self.x_size + self.extraSteering + lowest_left_line.middle_x) / 2
                self.steering_val = middle_x - self.x_size / 2
                cv2.putText(self.raw_image, f"{lowest_left_line.middle_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(self.raw_image, (int(middle_x), 50), 5, (255, 0, 0), -1)
            #nur die rechte Linie ist vorhanden
            elif len(right_lines) > 0:
                right_lines.sort(key=lambda x: x.middle_y)
                lowest_right_line = right_lines[-1]
                lowest_right_line.left = False
                lowest_right_line.draw(self.raw_image, (0, 0, 255)) #draw the line in blue
                #diese formel kann man noch optimieren. die fehlende linie sollte links vom bildrand um extraSteering verschoben werden

                middle_x = (lowest_right_line.middle_x - self.extraSteering ) / 2   
                cv2.putText(self.raw_image, f"{lowest_right_line.middle_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)   
                # middle_x = (self.x_size - self.extraSteering + lowest_right_line.middle_x) / 2          
                self.steering_val = middle_x - self.x_size / 2
                cv2.circle(self.raw_image, (int(middle_x), 50), 5, (255, 0, 0), -1)

            else:
                print("No lines detected")
