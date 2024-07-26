import os
from PIL import Image

# Path to the folder containing the PNG images
folder_path = 'PreCalcOut/datensatz/recordings_20240723_133750_kein_licht_an_kamera1'

# Get a list of all PNG files in the folder
png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

print(png_files)

# Benutzerdefinierte Sortierfunktion
def custom_sort(file_name):
    # Zerlege den Dateinamen am ersten "_"
    number_part = file_name.split(".", 1)[0]
    print(number_part)
    # Konvertiere den Teil vor
    return int(number_part)

png_files.sort(key=custom_sort)


# Create a list to store the image frames
frames = []

# Iterate over each PNG file
for file_name in png_files:
    print(file_name)
    # Open the image file
    image_path = os.path.join(folder_path, file_name)
    image = Image.open(image_path)

    # Append the image to the frames list
    frames.append(image)

# Save the frames as a GIF
gif_path = f'{folder_path}/output_128.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=20, loop=0)

print(f'GIF created successfully at {gif_path}')
