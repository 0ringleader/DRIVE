import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
import math
import json
import csv

# Define device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(32 * 25, 128)  # Adjust input size according to output size of the conv layers
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        return out

# Initialize the model
model = ConvNet(num_classes=4).to(device)
model.load_state_dict(torch.load('alles/cnn_track_04_5loops_v2.pth'))
model.eval()

pygame.init()

screen_width = 1200
screen_height = 900

screen = pygame.display.set_mode((screen_width, screen_height))
player = pygame.Rect(50, 30, 10, 10)
finish = pygame.Rect(120, 10, 2, 50)

car_angle = 0
car_speed = 2
crash_counter = 0

def line_intersects_rect(rect, start, end):
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

def line_intersects_line(start1, end1, start2, end2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = start1, end1
    C, D = start2, end2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def load_track_data(file_name='alles/track_data_03.json'):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading track data: {e}")
        return []
    
def define_line(line_start, line_end):
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

def get_front_direction(angle):
    rad_angle = math.radians(angle)
    return math.cos(rad_angle), -math.sin(rad_angle)

def get_control_data():
    keys = pygame.key.get_pressed()
    return [
        int(keys[pygame.K_w]),
        int(keys[pygame.K_a]),
        int(keys[pygame.K_s]),
        int(keys[pygame.K_d])
    ]
    
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
                
                if 0 <= x < screen_width and 0 <= y < screen_height:
                    color = screen.get_at((x, y))
                    if color == (0, 255, 0):  # Grün für die Streckenbegrenzung
                        distance = calculate_distance((player_center[0], player_center[1]), (x, y))
                        return distance
    return float(150)  # Maximaler Abstand

lines = load_track_data()

all_data = []
finish_line_crossings = 0

# Vor dem while-loop
distance_history = []

run = True
clock = pygame.time.Clock()
while run:
    clock.tick(30)
    screen.fill((0, 0, 0))
    
    for track_section in lines:
        if len(track_section) > 1:
            for i in range(len(track_section) - 1):
                pygame.draw.line(screen, (0, 255, 0), track_section[i], track_section[i + 1], 4)
    
    pygame.draw.rect(screen, (255, 0, 0), player)
    pygame.draw.rect(screen, (255, 192, 203), finish)
    
    player_center_x = player.x + player.width // 2
    player_center_y = player.y + player.height // 2
    player_center = (player_center_x, player_center_y)
    
    front_direction = get_front_direction(car_angle)
    line_start = (player_center[0], player_center[1])
    line_end = (line_start[0] + 150 * front_direction[0], line_start[1] + 150 * front_direction[1])
    pygame.draw.line(screen, (0, 0, 255), line_start, line_end, 2)
    
    right_vision = get_front_direction(car_angle + 60)
    right_end = (line_start[0] + 150 * right_vision[0], line_start[1] + 150 * right_vision[1])
    pygame.draw.line(screen, (0, 255, 255), line_start, right_end, 2)
    
    left_vision = get_front_direction(car_angle - 60)
    left_end = (line_start[0] + 150 * left_vision[0], line_start[1] + 150 * left_vision[1])
    pygame.draw.line(screen, (0, 255, 255), line_start, left_end, 2)

    right_vision_30 = get_front_direction(car_angle + 30)
    right_end_30 = (line_start[0] + 150 * right_vision_30[0], line_start[1] + 150 * right_vision_30[1])
    pygame.draw.line(screen, (255, 255, 0), line_start, right_end_30, 2)

    left_vision_30 = get_front_direction(car_angle - 30)
    left_end_30 = (line_start[0] + 150 * left_vision_30[0], line_start[1] + 150 * left_vision_30[1])
    pygame.draw.line(screen, (255, 255, 0), line_start, left_end_30, 2)
    
    distance_front = getDistance(car_angle, player_center, 150, vision_width=2)
    distance_left = getDistance(car_angle + 60, player_center, 150, vision_width=2)
    distance_right = getDistance(car_angle - 60, player_center, 150, vision_width=2)
    distance_right_30 = getDistance(car_angle + 30, player_center, 150, vision_width=2)
    distance_left_30 = getDistance(car_angle - 30, player_center, 150, vision_width=2)
    
    distances = [distance_front, distance_left, distance_right, distance_right_30, distance_left_30]
    distance_history.append(distances)
    if len(distance_history) > 10:
        distance_history.pop(0)

    # Debugging-Ausgabe der letzten 10 Abstandsdaten
    if len(distance_history) == 10:
        print("Last 10 distances:")
        for d in distance_history:
            print(d)

    # Nur vorhersagen, wenn wir genügend historische Daten haben
    if len(distance_history) == 10:
        distances_array = np.array(distance_history).flatten()  # Convert to numpy array and flatten
        distances_tensor = torch.tensor(distances_array, dtype=torch.float32).to(device).view(1, 1, -1)
        
        with torch.no_grad():
            outputs = model(distances_tensor)
            predicted = (outputs > 0.5).float().cpu().numpy().flatten()  # Convert logits to binary predictions
        
        # Logging der Modellvorhersage
        print(f'Control: {predicted}, Distances: {distances}')
        
        if predicted[0] == 1:
            player.x += car_speed * front_direction[0]
            player.y += car_speed * front_direction[1]
        if predicted[1] == 1:
            car_angle += 2
        if predicted[2] == 1:
            player.x -= car_speed * front_direction[0]
            player.y -= car_speed * front_direction[1]
        if predicted[3] == 1:
            car_angle -= 2
    
    key = pygame.key.get_pressed()
    if key[pygame.K_a]:
        car_angle += 2
    elif key[pygame.K_d]:
        car_angle -= 2
        
    if key[pygame.K_w]:
        player.x += car_speed * front_direction[0]
        player.y += car_speed * front_direction[1]
    elif key[pygame.K_s]:
        player.x -= car_speed * front_direction[0]
        player.y -= car_speed * front_direction[1]
        
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

    pygame.display.update()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()
