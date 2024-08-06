import requests
import cv2
from time import sleep
import numpy as np

class CarControl:
    def __init__(self, ip, port, stream_url):
        self.ip = ip
        self.port = port
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url)
        self.streaming = False
        print("CarControl initialized")

    def setControlData(self, speed, steering):
        data = {
            'speed': speed,
            'angle': steering
        }
        url = f"http://{self.ip}:{self.port}"  # Ändere dies zur tatsächlichen URL
        try:
            response = requests.post(url, json=data)
            print('Received response:', response.text)
        except Exception as e:
            print('Failed to send control data:', e)

    def close(self):
        self.cap.release()
        print("Connection closed")

    def start_stream(self, process_fn):
        self.process_fn = process_fn
        self.streaming = True
        print("Streaming started")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def stop_stream(self):
        self.streaming = False
        print("Streaming stopped")