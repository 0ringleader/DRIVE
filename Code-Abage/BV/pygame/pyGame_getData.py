import pygame
import sys
import math
import json
import csv

pygame.init()  # Initialize Pygameww

# Define the window size
screen_width = 1200
screen_height = 900

screen = pygame.display.set_mode((screen_width, screen_height))
player = pygame.Rect(50, 30, 10, 10)  # The player's rectangle
finish = pygame.Rect(120, 10, 2, 50)

# Initialize car variables
car_angle = 0  # Initial angle of the car
car_speed = 2  # Speed of the car
crash_counter = 0

# Function to save vision and control data
def save_data(data, filename='data_track_04.csv'):
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["distance_front", "distance_left", "distance_right", "distance_right_30", "distance_left_30", "w", "a", "s", "d"])  # Header row
            writer.writerows(data)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Failed to save data: {str(e)}")

# Function to load track data from a JSON file
def load_track_data(file_name='alles/track_data_04.json'):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading track data: {e}")
        return []

def define_line(line_start, line_end):
    # Bresenham's Line Algorithm to define the line
    points = []
    x1, y1 = line_start
    x2, y2 = line_end
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if (x1 == x2) and (y1 == y2):
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

# Define a function to get the front direction of the car
def get_front_direction(angle):
    rad_angle = math.radians(angle)
    return math.cos(rad_angle), -math.sin(rad_angle)

# Define a function to capture the control data
def get_control_data():
    keys = pygame.key.get_pressed()
    return [
        int(keys[pygame.K_w]),
        int(keys[pygame.K_a]),
        int(keys[pygame.K_s]),
        int(keys[pygame.K_d])
    ]

def line_intersects_rect(rect, start, end):
    """ Check if a line segment intersects with a rectangle. """
    rect_lines = [
        ((rect.left, rect.top), (rect.right, rect.top)),
        ((rect.right, rect.top), (rect.right, rect.bottom)),
        ((rect.right, rect.bottom), (rect.left, rect.bottom)),
        ((rect.left, rect.bottom), (rect.left, rect.top))
    ]
    for rect_start, rect_end in rect_lines:
        if line_intersects_line(rect_start, rect_end, start, end):
            return True
    return False

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def getDistance(car_angle, player_center, vision_length, vision_width=1):
    directions = [get_front_direction(car_angle)]
    for direction in directions:
        for i in range(1, vision_length + 1):
            for w in range(-vision_width, vision_width + 1):
                x = int(player_center[0] + i * direction[0] + w * direction[1])
                y = int(player_center[1] + i * direction[1] - w * direction[0])
                
                # Check if coordinates are within the screen boundaries
                if 0 <= x < screen_width and 0 <= y < screen_height:
                    color = screen.get_at((x, y))  # Get the color of the pixel at (x, y)
                    if color == (0, 255, 0):  # Check for green color
                        distance = calculate_distance((player_center[0], player_center[1]), (x, y))
                        return distance
    return float (150)  # Return a very large distance if no obstacle is found

def line_intersects_line(start1, end1, start2, end2):
    """ Check if two line segments intersect. """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = start1, end1
    C, D = start2, end2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Load the track data once before the game loop starts
lines = load_track_data()

# List to store the combined data
all_data = []
finish_line_crossings = 0  # Tracks how many times the car crosses the finish line

# Event Loop
run = True
clock = pygame.time.Clock()  # To control the frame rate
while run:
    clock.tick(50)
    screen.fill((0, 0, 0))  # Clear the screen
    
    # Draw the loaded track sections
    for track_section in lines:
        if len(track_section) > 1:
            for i in range(len(track_section) - 1):
                pygame.draw.line(screen, (0, 255, 0), track_section[i], track_section[i + 1], 4)
    
    pygame.draw.rect(screen, (255, 0, 0), player)
    pygame.draw.rect(screen, (255, 192, 203), finish)

    # Calculate the center of the player's rectangle
    player_center_x = player.x + player.width // 2
    player_center_y = player.y + player.height // 2
    
    # Create a point for the player's center
    player_center = (player_center_x, player_center_y)

    # Check for collision with the finish line
    if pygame.Rect(player_center_x, player_center_y, 1, 1).colliderect(finish):
        finish_line_crossings += 1    

    if finish_line_crossings == 6:  # Checks if this is the second crossing
        save_data(all_data)  # Save the combined data
        print("Second crossing! Data saved.")
        finish_line_crossings = 0  # Reset the counter if needed (or remove this if you only need to save once)
        
    ###############    
    
    # Get the front direction of the car
    front_direction = get_front_direction(car_angle)
    line_start = (player_center[0], player_center[1])
    line_end = (line_start[0] + 150 * front_direction[0], line_start[1] + 150 * front_direction[1])
    pygame.draw.line(screen, (0, 0, 255), line_start, line_end, 2)
    
    # Draw lines to visualize the vision (right)
    right_vision = get_front_direction(car_angle + 60)
    right_end = (line_start[0] + 150 * right_vision[0], line_start[1] + 150 * right_vision[1])
    pygame.draw.line(screen, (0, 255, 255), line_start, right_end, 2)
    
    # Draw lines to visualize the vision (left)
    left_vision = get_front_direction(car_angle - 60)
    left_end = (line_start[0] + 150 * left_vision[0], line_start[1] + 150 * left_vision[1])
    pygame.draw.line(screen, (0, 255, 255), line_start, left_end, 2)

    # Draw lines to visualize the vision (right 30 degrees)
    right_vision_30 = get_front_direction(car_angle + 30)
    right_end_30 = (line_start[0] + 150 * right_vision_30[0], line_start[1] + 150 * right_vision_30[1])
    pygame.draw.line(screen, (255, 255, 0), line_start, right_end_30, 2)

    # Draw lines to visualize the vision (left 30 degrees)
    left_vision_30 = get_front_direction(car_angle - 30)
    left_end_30 = (line_start[0] + 150 * left_vision_30[0], line_start[1] + 150 * left_vision_30[1])
    pygame.draw.line(screen, (255, 255, 0), line_start, left_end_30, 2)
    
    distance_front = getDistance(car_angle, player_center, 150, vision_width=2)  # 120 ist hier die vision_length
    distance_left = getDistance(car_angle + 60, player_center, 150, vision_width=2)
    distance_right = getDistance(car_angle - 60, player_center, 150, vision_width=2)
    distance_right_30 = getDistance(car_angle + 30, player_center, 150, vision_width=2)
    distance_left_30 = getDistance(car_angle - 30, player_center, 150, vision_width=2)
    
    distances = [distance_front, distance_left, distance_right, distance_right_30, distance_left_30]
    
    # Get control data
    control_data = get_control_data()
    
    # Handle keyboard input
    key = pygame.key.get_pressed()
    if key[pygame.K_a]:
        car_angle += 2  # Rotate left
    elif key[pygame.K_d]:
        car_angle -= 2  # Rotate right
        
    # Move the car forward based on its front direction
    if key[pygame.K_w]:
        player.x += car_speed * front_direction[0]
        player.y += car_speed * front_direction[1]
    elif key[pygame.K_s]:
        player.x -= car_speed * front_direction[0]
        player.y -= car_speed * front_direction[1]
        
    """
    # Self-driving car?
    if distance_front < 1:
        car_angle += 2
    if distance_left < 8:
        if distance_left < distance_right:
            car_angle -= 2  # Rotate right 
            control_data = [0,0,0,1]
        elif distance_right < distance_left:
            car_angle += 2  # Rotate left 
            control_data = [0,1,0,0]
    elif distance_right < 8:
        if distance_left < distance_right:
            car_angle -= 2
            control_data = [0,0,0,1]  
        elif distance_right < distance_left:
            car_angle += 2
            control_data = [0,1,0,0]  
    elif distance_right_30 < 10:
        car_angle -= 2
        control_data = [0,0,0,1]  
    elif distance_left_30 < 10:
        car_angle += 2
        control_data = [0,1,0,0]  
    else:
        player.x += car_speed * front_direction[0]
        player.y += car_speed * front_direction[1]
        control_data = [1,0,0,0]
    """
    
    # Combine distance and control data
    if control_data != [0,0,0,0]:
        combined_data = distances + control_data
        all_data.append(combined_data) 

    
    # Crash detection
    hit = False
    for line in lines:
        if len(line) > 1:
            for i in range(len(line) - 1):
                if line_intersects_rect(player, line[i], line[i + 1]):
                    hit = True
                    break
        if hit:
            break
    
    if hit:
        pygame.draw.rect(screen, (255, 255, 255), player)
    else:
        pygame.draw.rect(screen, (255, 0, 0), player)      

    pygame.display.update()  # Update the display
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()  # Quit the game
