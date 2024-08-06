import pygame
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import threading
import numpy as np

# Boolean variables to control plotting
PLOT_CONTROLLER_INPUTS = False # Set True to enable live-plotting
DETECT_D_PAD_INPUTS = False  # Set to True to include D-Pad inputs

# Initialize Pygame and joystick
pygame.init()
pygame.joystick.init()

# Create a file to save the joystick data
filename = "xbox_controller_data.csv"
file = open(filename, "w", newline='')
csv_writer = csv.writer(file)

# Write configuration line
csv_writer.writerow(["DETECT_D_PAD_INPUTS", DETECT_D_PAD_INPUTS])

# Check for controller
if pygame.joystick.get_count() == 0:
    print("No joystick detected. Please connect an Xbox controller and try again.")
    exit()

# Get the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Controller initialized: ", joystick.get_name())

# Get the number of axes and buttons
num_axes = joystick.get_numaxes()
num_buttons = joystick.get_numbuttons()

# Button names for Xbox controller
button_names = ["A", "B", "X", "Y", "LB", "RB", "Back", "Start", "LStick", "RStick"]

# Prepare CSV header
csv_header = ["Timestamp"] + [f"Axis_{i}" for i in range(num_axes)]
if DETECT_D_PAD_INPUTS:
    csv_header += [f"DPad_{i}" for i in range(4)]
csv_header += [f"Button_{button_names[i]}" if i < len(button_names) else f"Button_{i}" for i in range(num_buttons)]
csv_writer.writerow(csv_header)

if PLOT_CONTROLLER_INPUTS:
    # Data storage for plotting
    time_window = 10  # seconds
    update_interval = 0.05  # 50 ms
    data_points = int(time_window / update_interval)

    timestamps = np.linspace(0, time_window, data_points)
    left_stick = [np.zeros(data_points), np.zeros(data_points)]
    right_stick = [np.zeros(data_points), np.zeros(data_points)]
    shoulder_buttons = [np.zeros(data_points), np.zeros(data_points)]
    buttons_data = [np.zeros(data_points) for _ in range(num_buttons)]
    dpad_data = [np.zeros(data_points) for _ in range(4)] if DETECT_D_PAD_INPUTS else None

    # Setup plots
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4 + (1 if DETECT_D_PAD_INPUTS else 0), 1, figsize=(10, 20),
                                             num="Controller Inputs")

    left_lines = [ax1.plot([], [], label=f'Left Stick {"X" if i == 0 else "Y"}')[0] for i in range(2)]
    right_lines = [ax2.plot([], [], label=f'Right Stick {"X" if i == 0 else "Y"}')[0] for i in range(2)]
    shoulder_lines = [ax3.plot([], [], label=f'{"LT" if i == 0 else "RT"}')[0] for i in range(2)]
    button_lines = [ax4.plot([], [], label=f'{button_names[i] if i < len(button_names) else f"Button_{i}"}')[0] for i in
                    range(num_buttons)]

    if DETECT_D_PAD_INPUTS:
        dpad_lines = [fig.add_subplot(5, 1, 5).plot([], [], label=f'DPad_{i}')[0] for i in range(4)]

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, time_window)
        ax.set_ylim(-1, 1)  # Set Y-axis limits to -1 and 1
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Reset Y-axis limits for button and D-Pad plots
    ax4.set_ylim(0, 1)  # Set Y-axis limits for buttons plot

    if DETECT_D_PAD_INPUTS:
        ax5 = fig.add_subplot(5, 1, 5)
        ax5.set_xlim(0, time_window)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3)
        ax5.set_title('D-Pad')
        ax5.set_ylim(0, 1)  # Set Y-axis limits for D-Pad plot
        for i in range(4):
            dpad_lines[i].set_label(f'DPad_{i}')

    ax1.set_title('Left Joystick')
    ax2.set_title('Right Joystick')
    ax3.set_title('Shoulder Buttons (Triggers)')
    ax4.set_title('Buttons')

    # Add button to stop the program
    ax_button = plt.axes([0.45, 0.95, 0.1, 0.05])  # [left, bottom, width, height]
    button = Button(ax_button, 'Stop')


    def stop_program(event):
        print("Stopping...")
        stop_event.set()
        plt.close('all')  # Close all figures


    button.on_clicked(stop_program)

start_time = time.time()
lock = threading.Lock()
stop_event = threading.Event()


def update_data():
    global left_stick, right_stick, shoulder_buttons, buttons_data, dpad_data
    while not stop_event.is_set():
        # Get joystick data
        pygame.event.pump()  # Process event queue

        # Get all axis values
        axes = [joystick.get_axis(i) for i in range(num_axes)]

        # Get all button values
        buttons = [joystick.get_button(i) for i in range(num_buttons)]  # Binary inputs 0 or 1

        # Get D-Pad values if enabled
        dpad = [joystick.get_hat(i) for i in range(1)] if DETECT_D_PAD_INPUTS else [0, 0, 0, 0]

        with lock:
            if PLOT_CONTROLLER_INPUTS:
                # Update data arrays
                for i in range(2):
                    left_stick[i] = np.roll(left_stick[i], -1)
                    left_stick[i][-1] = axes[i]
                    right_stick[i] = np.roll(right_stick[i], -1)
                    right_stick[i][-1] = axes[i + 2]
                    shoulder_buttons[i] = np.roll(shoulder_buttons[i], -1)
                    shoulder_buttons[i][-1] = axes[i + 4]

                for i in range(num_buttons):
                    buttons_data[i] = np.roll(buttons_data[i], -1)
                    buttons_data[i][-1] = buttons[i]

                if DETECT_D_PAD_INPUTS:
                    for i in range(4):
                        dpad_data[i] = np.roll(dpad_data[i], -1)
                        dpad_data[i][-1] = dpad[0][i] if i < len(dpad[0]) else 0

        # Write data to CSV file
        csv_writer.writerow(
            [time.time() - start_time] + axes + (dpad if DETECT_D_PAD_INPUTS else [0, 0, 0, 0]) + buttons)

        time.sleep(0.01)  # 100 Hz data collection


def update_plot(frame):
    current_time = time.time() - start_time
    with lock:
        # Update plot data
        for i, line in enumerate(left_lines):
            line.set_data(timestamps + current_time, left_stick[i])
        for i, line in enumerate(right_lines):
            line.set_data(timestamps + current_time, right_stick[i])
        for i, line in enumerate(shoulder_lines):
            line.set_data(timestamps + current_time, shoulder_buttons[i])
        for i, line in enumerate(button_lines):
            line.set_data(timestamps + current_time, buttons_data[i])

        if DETECT_D_PAD_INPUTS:
            for i, line in enumerate(dpad_lines):
                line.set_data(timestamps + current_time, dpad_data[i])

        # Adjust x-axis limits to create scrolling effect
        start = current_time
        end = current_time + time_window

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(start, end)
            ax.set_xticks(np.linspace(start, end, 6))
            ax.set_xticklabels([f'{t:.1f}' for t in np.linspace(start, end, 6)])

        ax4.set_xlim(start, end)
        ax4.set_xticks(np.linspace(start, end, 6))
        ax4.set_xticklabels([f'{t:.1f}' for t in np.linspace(start, end, 6)])

        if DETECT_D_PAD_INPUTS:
            ax5.set_xlim(start, end)
            ax5.set_xticks(np.linspace(start, end, 6))
            ax5.set_xticklabels([f'{t:.1f}' for t in np.linspace(start, end, 6)])

    return left_lines + right_lines + shoulder_lines + button_lines + (dpad_lines if DETECT_D_PAD_INPUTS else [])


# Start data collection in a separate thread
data_thread = threading.Thread(target=update_data, daemon=True)
data_thread.start()

if PLOT_CONTROLLER_INPUTS:
    # Create animation
    ani = FuncAnimation(fig, update_plot, frames=None, interval=50, blit=True, cache_frame_data=False)
    plt.show()
else:
    try:
        while data_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

# Stop data collection and cleanup
stop_event.set()
data_thread.join()
file.close()
pygame.quit()






