#!/usr/bin/python3

import os
import io
import logging
import socketserver
from http import server
from threading import Condition
import json
import time

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from MotorControl import ArduinoControl

#Control Motors
arduino = ArduinoControl('/dev/ttyUSB0')  # Replace with your Arduino's port
arduino.set_motor_speed(0)
arduino.set_servo_angle(90)
car_speed = 0
car_angle = 90

# Step 1: Define the paths to control.html, joy.js, and style.css
CONTROL_HTML_PATH = os.path.join(os.getcwd(), 'control.html')
JOY_JS_PATH = os.path.join(os.getcwd(), 'joy.js')
STYLE_CSS_PATH = os.path.join(os.getcwd(), 'style.css')

# Step 3: Read the content of control.html
with open(CONTROL_HTML_PATH, 'r') as file:
    PAGE = file.read()

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve control.html as the root page
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(PAGE))
            self.end_headers()
            self.wfile.write(PAGE.encode('utf-8'))
        elif self.path == '/joy.js':
            # Serve joy.js
            with open(JOY_JS_PATH, 'r') as js_file:
                js_content = js_file.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript')
            self.send_header('Content-Length', len(js_content))
            self.end_headers()
            self.wfile.write(js_content.encode('utf-8'))
        elif self.path == '/style.css':
            # Serve style.css
            with open(STYLE_CSS_PATH, 'r') as css_file:
                css_content = css_file.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/css')
            self.send_header('Content-Length', len(css_content))
            self.end_headers()
            self.wfile.write(css_content.encode('utf-8'))
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()
    
    def do_POST(self):
        # Handle POST requests
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        decoded_post_data = post_data.decode('utf-8')
        
        try:
            data = json.loads(decoded_post_data)
            if 'speed' in data:
                arduino.set_motor_speed(data['speed'])
            if 'angle' in data:
                angle = int(data['angle'])*0.5 + 90 
                arduino.set_servo_angle(angle)
                print(angle)
                
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Invalid JSON')
            return

        # Send a response back to the client indicating success
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Successful POST request received.')

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()
