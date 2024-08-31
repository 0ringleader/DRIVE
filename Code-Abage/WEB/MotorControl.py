import serial
import time

class ArduinoControl:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
        time.sleep(2)  # Wait for the Arduino to reset

    def set_motor_speed(self, speed):
        if -100 <= speed <= 100:
            self.ser.write(b'S')
            self.ser.write(str(-speed).encode())
            self.ser.write(b'\n')

    def set_servo_angle(self, angle):
        if 40 <= angle <= 140:
            self.ser.write(b'A')
            self.ser.write(str(angle).encode())
            self.ser.write(b'\n')

    def get_current_sensor_values(self):
        self.ser.write(b'C')
        self.ser.timeout = 1  # Setzt einen Timeout von 1 Sekunde
        response = self.ser.readline().decode().strip()
        if response:  # Überprüft, ob eine Antwort erhalten wurde
            current_A, current_B = response.split(',')
            return int(current_A), int(current_B)
        else:
            print("Keine Antwort von der seriellen Schnittstelle erhalten.")
            return None

    def close(self):
        self.ser.close()