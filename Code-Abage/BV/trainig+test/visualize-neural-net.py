import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import logging
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Custom loss function if needed
def custom_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_error = tf.square(error)
    abs_error = tf.abs(error)
    loss = tf.where(abs_error < 1.0, squared_error, abs_error)
    return tf.reduce_mean(loss)

# Path to your saved model
model_path = "/home/terrox/Documents/johnnysCode/Alex-ai/old model/lane_detection_model (1).keras"

# Load the trained model
model = load_model(model_path, custom_objects={"custom_loss": custom_loss})

# Visualize the model
visualization_path = '/home/terrox/Documents/johnnysCode/Alex-ai/model_visualization_old70MB.png'
plot_model(model, to_file=visualization_path, show_shapes=True, show_layer_names=True)

# Inform user where the file is saved
print(f'Model visualization saved to: {visualization_path}')

# Display the image using PIL
from PIL import Image

image = Image.open(visualization_path)
image.show()