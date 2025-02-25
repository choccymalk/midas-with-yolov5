import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="depth_anything_v2_edgetpu.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))  # Resize to the expected input size
img = img.astype(np.float32) / 255.0  # Normalize pixel values
input_data = np.expand_dims(img, axis=0)

# Set input tensor and invoke the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output depth map
depth_map = interpreter.get_tensor(output_details[0]['index'])[0]

# Post-process the depth map (example: normalize and convert to grayscale)
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_map = (depth_map * 255).astype(np.uint8)
depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

# Display or save the depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
