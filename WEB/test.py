from MotorControl import ArduinoControl

arduino = ArduinoControl('/dev/ttyUSB0')  # Replace with your Arduino's port
arduino.set_motor_speed(0)
arduino.set_servo_angle(90)

while True:
    what_to_do = input('What do you want to do? 1: Set motor speed, 2: Set servo angle, 3: Get current sensor values, 4: Exit\n')
    if what_to_do == '1':
        speed = int(input('Enter the speed: '))
        arduino.set_motor_speed(speed)
    elif what_to_do == '2':
        angle = int(input('Enter the angle: '))
        arduino.set_servo_angle(angle)
    elif what_to_do == '3':
        print(arduino.get_current_sensor_values())
    elif what_to_do == '4':
        arduino.close()
        break
    else:
        print('Invalid input. Please try again.')
        0