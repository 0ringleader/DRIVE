import requests
import cv2
from time import sleep
import numpy as np

class CarControl:
    """
    A class that represents a car control system.

    Attributes:
        ip (str): The IP address of the car control system.
        port (int): The port number of the car control system.
        stream_url (str): The URL of the video stream.
        cap (cv2.VideoCapture): The video capture object.
        streaming (bool): Indicates whether the video streaming is active.

    Methods:
        __init__(self, ip, port, stream_url): Initializes the CarControl object.
        setControlData(self, speed, steering): Sends control data to the car control system.
        close(self): Closes the connection to the car control system.
        read_frame(self): Reads a frame from the video stream.
    """

    def __init__(self, ip, port, stream_url):
        """
        Initializes the CarControl object.

        Args:
            ip (str): The IP address of the car control system.
            port (int): The port number of the car control system.
            stream_url (str): The URL of the video stream.
        """
        self.ip = ip
        self.port = port
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url)
        print("CarControl initialized")

    def setControlData(self, speed, steering):
        """
        Sends control data to the car control system.

        Args:
            speed (float): The speed value for the car control.
            steering (float): The steering angle value for the car control.
        """
        data = {
            'speed': float(speed),
            'angle': float(steering)
        }
        url = f"http://{self.ip}:{self.port}"  # Change this to the actual URL
        try:
            response = requests.post(url, json=data)
            #print('Received response:', response.text)
        except Exception as e:
            print('Failed to send control data:', e)


    def close(self):
        """
        Closes the connection to the car control system.
        """
        self.cap.release()
        print("Connection closed")

    def read_frame(self):
        """
        Reads a frame from the video stream.

        Returns:
            frame (numpy.ndarray): The frame read from the video stream.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None

        return frame