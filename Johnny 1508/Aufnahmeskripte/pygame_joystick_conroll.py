import cv2
import numpy as np
import pygame
from CarControl import CarControl
import os
import datetime


# Globale Variablen für den aktuellen Zustand
current_speed = 0
current_angle = 50


max_speed = 60


#car_control = CarControl('192.168.1.162', 8000, 'http://192.168.1.162:8000/stream')
car_control = CarControl('10.42.0.1', 8000, 'http://10.42.0.1:8000/stream')
# car_control = CarControl('10.42.0.186', 8000, 'http://10.42.0.1:8000/stream')
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()


try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                # Maximalgeschwindigkeit anpassen
                if joystick.get_button(11):  # Taste nach oben
                    max_speed = min(100, max_speed + 10)
                elif joystick.get_button(12):  # Taste nach unten
                    max_speed = max(0, max_speed - 10)
                

                        

        speed= 0
        angle = 0

        # Vorwärts/Rückwärts fahren
        r2_value = joystick.get_axis(5)  # R2-Taste (Trigger)
        l2_value = joystick.get_axis(4)  # L2-Taste (Trigger)

        if r2_value > 0:
            speed = int(r2_value * max_speed)
        elif l2_value > 0:
            speed = int(-l2_value * max_speed)

        # Lenkung
        axis_value = joystick.get_axis(2)  # Rechter Stick horizontal
        angle = int(axis_value * 100)  # Wert von -100 bis 100
        
        # Überprüfen, ob eine Änderung stattgefunden hat
        if speed != current_speed or angle != current_angle:
            car_control.setControlData(speed, angle)
            current_speed = speed
            current_angle = angle
            print("CONTROLL")
except KeyboardInterrupt:
    pass

car_control.close()
pygame.quit()