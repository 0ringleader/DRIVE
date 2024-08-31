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

        #calculate middle
        self.middle_x = (x1 + x2) / 2
        self.middle_y = (y1 + y2) / 2

        

    def __str__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Middle: ({self.middle_x}, {self.middle_y})"

    def __repr__(self):
        return f"Line: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}), Slope: {self.slope}, Middle: ({self.middle_x}, {self.middle_y})"
    def draw(self, image = None, color = (0, 255, 0)):      
        # Zeichne den gedrehten Pfeil durch Vertauschen der Punkte
        cv2.arrowedLine(image, (int(self.x2), int(self.y2)), (int(self.x1), int(self.y1)), color, 2, tipLength=0.1)
    
    
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
        self.offset = 50 #wenn der alogrhitmus nur eine linie erkennt wird die andere seite 50 pixel nach außen geschoben

    def preprocess_frame(self, image):
        self.raw_image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(blurred)

        edges = cv2.Canny(cl1, 150, 200)

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


    def get_steering_val(self, draw_lines = False):
        left_lines = []
        right_lines = []
        lowest_left_line = None
        lowest_right_line = None
        middle_x = None

        for line in self.lines:
            if line.slope < 0:
                left_lines.append(line)
            elif line.slope > 0:
                right_lines.append(line)

        #herausschmeißen von linien die auf der falschen seite sind, um den algorithmus zu stabilisieren (optional)
        if len(left_lines) > 0 and len(right_lines) > 0:
            left_median_x = np.median([line.middle_x for line in left_lines])
            right_median_x = np.median([line.middle_x for line in right_lines])           
            left_lines = [line for line in left_lines if line.middle_x < right_median_x]
            right_lines = [line for line in right_lines if line.middle_x > left_median_x]

            #schmeße alle rechten linien auf der linken hälfte raus
            right_lines = [line for line in right_lines if line.middle_x > self.x_size / 2]
            #schmeiße alle linken linien auf der rechten hälfte raus
            left_lines = [line for line in left_lines if line.middle_x < self.x_size / 2]

        if len(left_lines) > 0:
            left_lines.sort(key=lambda x: x.middle_y)
            lowest_left_line = left_lines[-1]
        if len(right_lines) > 0:
            right_lines.sort(key=lambda x: x.middle_y)
            lowest_right_line = right_lines[-1]

        #fall 1 beide linien sind vorhanden
        if len(left_lines) > 0 and len(right_lines) > 0:
            if draw_lines:
                lowest_left_line.draw(self.raw_image, (0, 255, 0))
                lowest_right_line.draw(self.raw_image, (0, 0, 255))
            #x-Wert vom schnittpunkt der linken linie mit dem unteren Bildrand
            x_left = lowest_left_line.middle_x - (lowest_left_line.middle_y - self.y_size) / lowest_left_line.slope
            #x-Wert vom schnittpunkt der rechten linie mit dem unteren Bildrand
            x_right = lowest_right_line.middle_x - (lowest_right_line.middle_y - self.y_size) / lowest_right_line.slope
        #fall 2 nur eine linie ist vorhanden
        else:
            # Fall 2.1 nur die linke Linie ist vorhanden
            if len(left_lines) > 0:
                if draw_lines:
                    lowest_left_line.draw(self.raw_image, (0, 255, 0))
                #Berechne den x-Wert vom Schnittpunkt der linken Linie mit dem unteren Bildrand
                x_left = lowest_left_line.middle_x - (lowest_left_line.middle_y - self.y_size) / lowest_left_line.slope
                x_right = self.x_size + self.offset

                #fall 2.2 nur die rechte Linie ist vorhanden
            elif len(right_lines) > 0:
                if draw_lines:
                    lowest_right_line.draw(self.raw_image, (0, 0, 255))
                #Berechne den x-Wert vom Schnittpunkt der rechten Linie mit dem unteren Bildrand
                x_right = lowest_right_line.middle_x - (lowest_right_line.middle_y - self.y_size) / lowest_right_line.slope
                x_left = -self.offset
        #berechne den abstand zur bildmitte der schnittpunkte
        try: 
            middle_x = (x_left + x_right) / 2
            self.steering_val = (middle_x - self.x_size / 2)
            #clamp steering value to -100, 100
            self.steering_val = min(100, max(-100, self.steering_val))
            cv2.circle(self.raw_image, (int(middle_x), 50), 5, (255, 0, 0), -1)
        except:
            print("No lines detected")
            pass