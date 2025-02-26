import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
#depth_anything = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': 20})
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="depth_anything_v2_vits_indoor_dynamic_full_integer_quant.tflite")
# model input tensor should be tensor: int8[1,518,518,3]
# model output tensor should be tensor: int8[1,518,518]
# Allocate tensors for the model
interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
# Get input and output tensor details
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
# Load and preprocess the image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (518, 518))  # Resize to the expected input size
img = img.astype(np.int8) / 255.0  # Normalize pixel values
input_data = np.expand_dims(img, axis=0)

# Set input tensor and invoke the interpreter
# convert to int8 from float32 to int8 for Coral TPU
input_data = (input_data * 255).astype(np.int8)
interpreter.set_tensor(input_details[0]['index'], input_data)
# set max depth
#interpreter.set_tensor(input_details[0]['max_depth'], np.array([20], dtype=np.int8))
# set encoder
#interpreter.set_tensor(input_details[0]['encoder'], 'vits'.encode())
start_time = time.time()
# pass arguments to model
interpreter.invoke()
# Get the output tensor and post-process it (example: convert to grayscale)


# Get the output depth map
depth_map = interpreter.get_tensor(output_details[0]['index'])[0]
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
# Post-process the depth map (example: normalize and convert to grayscale)
#depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_map = depth_map.astype(np.uint8)
print(depth_map)
#depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)  # Optional: apply a color map for visualization
#depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
cmap = matplotlib.colormaps.get_cmap('gnuplot2')

depth_map = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

# Display or save the depth map
cv2.imwrite("Depth-Map.jpg", depth_map)
# print inference time in ms
end_time = time.time()
inference_time_ms = (end_time - start_time) * 1000
print(f"Inference Time: {inference_time_ms:.2f} ms")
