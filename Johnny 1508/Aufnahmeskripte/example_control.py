import pygame
from CarControl import CarControl
import time

def process_frame(frame):
    print("Frame erhalten")
    pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
    pygame.display.flip()
    pygame.display.update()
    if pygame.key.get_pressed()[pygame.K_q]:
        car.stop_stream()
        pygame.quit()
        exit()
    

pygame.init()
screen = pygame.display.set_mode((640, 480))
car = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')
while True:
    frame = car.read_frame()
    if frame is not None:
        process_frame(frame)
    else:
        break

car.stop_stream()

pygame.quit()