import serial.tools.list_ports

def scan_usb_devices():
    # List all connected ports
    ports = list(serial.tools.list_ports.comports())
    
    # Iterate through the list of ports
    for port in ports:
        # Print the device name and serial number if available
        print(f"Device Name: {port.device}, Serial Number: {port.serial_number}")

# Call the function to scan and display USB devices
scan_usb_devices()
