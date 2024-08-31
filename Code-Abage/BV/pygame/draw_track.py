import pygame
import sys
import math
import json

pygame.init()

# Variables defining the size of the window
screen_width = 1200
screen_height = 900

screen = pygame.display.set_mode((screen_width, screen_height))

# List to store lines
lines = []
current_line = []
version = 1

# Define the saving function
def save_track_as_json(lines):
    try:
        with open('track_data.json', 'w') as f:
            json.dump(lines, f)
        print("Track saved successfully.")
    except Exception as e:
        print(f"Failed to save track: {str(e)}")

# Create a font object
font = pygame.font.Font(None, 24)

# Create a surface for the button
button_surface = pygame.Surface((50, 25))

# Render text on the button
text = font.render("Click Me", True, (0, 0, 0))
text_rect = text.get_rect(center=(button_surface.get_width()/2, button_surface.get_height()/2))

# Create a pygame.Rect object that represents the button's boundaries
button_rect = pygame.Rect(60, 35, 60, 35)  # Adjust the position as needed

# Event Loop
run = True
clock = pygame.time.Clock()

while run:
    clock.tick(90)
    
    screen.fill((0, 0, 0))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                current_line = [event.pos]
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and current_line:  # Left mouse button
                current_line.append(event.pos)
                lines.append(current_line)
                current_line = []
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:  # Left mouse button held down
                current_line.append(event.pos)
        
        # Check if the button was clicked
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and button_rect.collidepoint(event.pos):
            save_track_as_json(lines)
            print("Button clicked!")

    # Draw the button
    if button_rect.collidepoint(pygame.mouse.get_pos()):
        pygame.draw.rect(button_surface, (127, 255, 212), (1, 1, 148, 48))
    else:
        pygame.draw.rect(button_surface, (0, 0, 0), (0, 0, 150, 50))
        pygame.draw.rect(button_surface, (255, 255, 255), (1, 1, 148, 48))
        pygame.draw.rect(button_surface, (0, 0, 0), (1, 1, 148, 1), 2)
        pygame.draw.rect(button_surface, (0, 100, 0), (1, 48, 148, 10), 2)
    
    button_surface.blit(text, text_rect)
    screen.blit(button_surface, (button_rect.x, button_rect.y))
    
    # Draw the current line being drawn
    if len(current_line) > 1:
        pygame.draw.lines(screen, (0, 255, 0), False, current_line, 4)
        
    # Redraw all previously drawn lines
    for line in lines:
        if len(line) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, line, 4)    
    
    pygame.display.update()

pygame.quit()
sys.exit()
