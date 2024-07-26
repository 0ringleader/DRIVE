#controller.py edited 3:50PM 2607

import pygame
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import threading
import numpy as np


class XboxController:
    def __init__(self, plot_controller_inputs=True):
        pygame.init()
        pygame.joystick.init()

        self.detect_d_pad_inputs = True
        self.plot_controller_inputs = plot_controller_inputs
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected. Please connect an Xbox controller and try again.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print("Controller initialized: ", self.joystick.get_name())

        self.num_axes = self.joystick.get_numaxes()
        self.num_buttons = self.joystick.get_numbuttons()

        self.filename = "xbox_controller_data.csv"
        self.file = open(self.filename, "w", newline='')
        self.csv_writer = csv.writer(self.file)

        self.csv_writer.writerow(
            ["Timestamp", "Left_Stick_X", "Left_Stick_Y", "Right_Stick_X", "Right_Stick_Y", "LT", "RT", "DPad_Up",
             "DPad_Right", "DPad_Down", "DPad_Left"])

        self.initialize_plot()

        self.start_time = time.time()

        self.data_thread = threading.Thread(target=self.update_data, daemon=True)
        self.data_thread.start()

    def initialize_plot(self):
        if self.plot_controller_inputs:
            self.time_window = 10  # seconds
            self.update_interval = 0.05  # 50 ms
            self.data_points = int(self.time_window / self.update_interval)

            self.timestamps = np.linspace(0, self.time_window, self.data_points)
            self.left_stick = [np.zeros(self.data_points), np.zeros(self.data_points)]
            self.right_stick = [np.zeros(self.data_points), np.zeros(self.data_points)]
            self.shoulder_buttons = [np.zeros(self.data_points), np.zeros(self.data_points)]
            self.dpad_data = [np.zeros(self.data_points) for _ in range(4)]

            plt.style.use('dark_background')

            self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
                4, 1, figsize=(10, 15), num="Controller Inputs"
            )

            self.left_lines = [self.ax1.plot([], [], label=f'Left Stick {"X" if i == 0 else "Y"}')[0] for i in range(2)]
            self.right_lines = [self.ax2.plot([], [], label=f'Right Stick {"X" if i == 0 else "Y"}')[0] for i in
                                range(2)]
            self.shoulder_lines = [self.ax3.plot([], [], label=f'{"LT" if i == 0 else "RT"}')[0] for i in range(2)]
            self.dpad_lines = [self.ax4.plot([], [],
                                             label=f'DPad_{"Up" if i == 0 else "Right" if i == 1 else "Down" if i == 2 else "Left"}')[
                                   0] for i in range(4)]

            for ax in (self.ax1, self.ax2, self.ax3):
                ax.set_xlim(0, self.time_window)
                ax.set_ylim(-1, 1)  # Set Y-axis limits to -1 and 1
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)

            # Reset Y-axis limits for D-Pad plot
            self.ax4.set_ylim(-1, 1)  # Set Y-axis limits for buttons plot

            self.ax1.set_title('Left Joystick')
            self.ax2.set_title('Right Joystick')
            self.ax3.set_title('Shoulder Buttons (Triggers)')
            self.ax4.set_title('D-Pad')

            # Add button to stop the program
            ax_button = plt.axes([0.45, 0.95, 0.1, 0.05])  # [left, bottom, width, height]
            button = Button(ax_button, 'Stop')
            button.on_clicked(self.stop_program)

            self.ani = FuncAnimation(self.fig, self.update_plot, frames=None, interval=50, blit=True,
                                     cache_frame_data=False)

    def stop_program(self, _event=None):
        print("Stopping...")
        self.stop_event.set()
        plt.close('all')  # Close all figures

    def update_data(self):
        while not self.stop_event.is_set():
            pygame.event.pump()

            axes = [self.joystick.get_axis(i) for i in range(self.num_axes)]
            dpad = list(self.joystick.get_hat(0)) if self.detect_d_pad_inputs else [0, 0, 0, 0]

            with self.lock:
                if self.plot_controller_inputs:
                    for i in range(2):
                        self.left_stick[i] = np.roll(self.left_stick[i], -1)
                        self.left_stick[i][-1] = axes[i]
                        self.right_stick[i] = np.roll(self.right_stick[i], -1)
                        self.right_stick[i][-1] = axes[i + 2]
                        self.shoulder_buttons[i] = np.roll(self.shoulder_buttons[i], -1)
                        self.shoulder_buttons[i][-1] = axes[i + 4]

                    for i in range(4):
                        self.dpad_data[i] = np.roll(self.dpad_data[i], -1)
                        self.dpad_data[i][-1] = dpad[i] if i < len(dpad) else 0

            self.csv_writer.writerow([time.time() - self.start_time] + axes[:4] + axes[4:6] + dpad)
            time.sleep(0.01)

    def update_plot(self, _frame):
        current_time = time.time() - self.start_time
        with self.lock:
            for i, line in enumerate(self.left_lines):
                line.set_data(self.timestamps + current_time - self.time_window, self.left_stick[i])
            for i, line in enumerate(self.right_lines):
                line.set_data(self.timestamps + current_time - self.time_window, self.right_stick[i])
            for i, line in enumerate(self.shoulder_lines):
                line.set_data(self.timestamps + current_time - self.time_window, self.shoulder_buttons[i])
            for i, line in enumerate(self.dpad_lines):
                line.set_data(self.timestamps + current_time - self.time_window, self.dpad_data[i])

            start = current_time - self.time_window
            end = current_time

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.set_xlim(start, end)
                ax.set_xticks(np.linspace(start, end, 6))
                ax.set_xticklabels([f'{t:.1f}' for t in np.linspace(start, end, 6)])

        return self.left_lines + self.right_lines + self.shoulder_lines + self.dpad_lines

    def start_plot(self):
        if self.plot_controller_inputs:
            plt.show()

    def cleanup(self):
        self.stop_event.set()
        self.data_thread.join()
        self.file.close()
        pygame.quit()

    def get_controller_inputs(self):
        with self.lock:
            axes = [self.joystick.get_axis(i) for i in range(self.num_axes)]
            dpad = list(self.joystick.get_hat(0)) if self.detect_d_pad_inputs else [0, 0]
        return axes[:4] + axes[4:6] + dpad


if __name__ == "__main__":
    controller = XboxController(plot_controller_inputs=True)
    try:
        controller.start_plot()
    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()