import time
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import glob
from transformers import pipeline
yolo_model = torch.hub.load(".", "custom", path="../yolov5s.pt", source="local", device="cpu")
CLASSES_TO_DETECT = [0]
CONFIDENCE_THRESHOLD = 0.7
def calculate_object_angle(bbox, depth_map, camera_params):
    """
    Calculate the object's angle relative to the camera center.
    
    Parameters:
    bbox (tuple): The object's bounding box coordinates (x1, y1, x2, y2)
    depth_map (numpy.ndarray): The depth map for the image
    camera_params (CameraParams): The camera calibration parameters
    
    Returns:
    tuple: The object's angle in the X and Y dimensions (angle_x_deg, angle_y_deg)
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate the object's center point
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Get the image center coordinates
    image_center_x = depth_map.shape[1] / 2
    image_center_y = depth_map.shape[0] / 2
    
    # Calculate the object's offset from the image center
    dx = cx - image_center_x
    dy = cy - image_center_y
    
    # Get the object's distance from the camera
    object_distance = depth_map[int(cy), int(cx)]
    
    # Calculate the angles in radians
    angle_x = np.arctan2(dx, object_distance)
    angle_y = np.arctan2(dy, object_distance)
    
    # Convert to degrees
    angle_x_deg = np.rad2deg(angle_x)
    angle_y_deg = np.rad2deg(angle_y)
    
    return angle_x_deg, angle_y_deg
class CameraCalibrator:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.board_size = (9, 6)  # Interior points of checkerboard
        self.square_size = 12.7  # mm

    def calibrate(self, images_path):
        """
        Calibrate camera using checkerboard images
        """
        # Prepare object points
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        images = glob.glob(os.path.join(images_path, '*.jpg'))
        
        if not images:
            raise ValueError(f"No calibration images found in {images_path}")

        img_shape = None
        for fname in images:
            img = cv2.imread(fname)
            img = cv2.resize(img,(int(512),int(512)))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
            if img_shape is None:
                img_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            print("line 46")
            if ret:
                obj_points.append(objp)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                img_points.append(refined_corners)

                # Draw and display the corners (optional)
                cv2.drawChessboardCorners(img, self.board_size, refined_corners, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)

        if not obj_points:
            raise ValueError("No valid calibration patterns found in images")

        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

        # Save calibration results
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'image_size': img_shape
        }
        np.save('camera_calibration.npy', calibration_data)
        
        return calibration_data

class CameraParams:
    def __init__(self, calibration_file=None):
        if calibration_file and os.path.exists(calibration_file):
            self.calibration_data = np.load(calibration_file, allow_pickle=True).item()
            self.camera_matrix = self.calibration_data['camera_matrix']
            self.dist_coeffs = self.calibration_data['dist_coeffs']
            self.focal_length_px = self.camera_matrix[0, 0]  # Assuming fx ≈ fy
        else:
            self.calibration_data = None
            self.camera_matrix = None
            self.dist_coeffs = None
            self.focal_length_px = None

def load_depth_anything_model(image_path):
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    image = Image.open(image_path)
    depth = pipe(image)["depth"]
    print(depth)

def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    midas_transforms = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, midas_transforms, device

def calculate_metric_depth(depth_map, camera_params, baseline_depth=1.0):  # Increased baseline
    """
    Convert relative depth to metric depth using camera parameters
    """
    if camera_params.focal_length_px is None:
        return depth_map  # Return raw depth map
        
    # Normalize depth map to 0-1 range
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply more conservative scaling
    scale_factor = baseline_depth / depth_map_norm.mean()
    scaled_depth = depth_map_norm * scale_factor
    
    # Apply reasonable bounds
    scaled_depth = np.clip(scaled_depth, 0, baseline_depth * 2)
    
    return scaled_depth

def apply_depth_refinement(depth_map):
    depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d=7, sigmaColor=0.1, sigmaSpace=5)
    kernel = np.ones((3,3), np.uint8)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    return depth_map

def get_color_by_confidence(confidence):
    if confidence > 0.8:
        return (0, 255, 0)
    elif confidence > 0.6:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def draw_detection(image, box, class_name, confidence, depth_value=None, angle_x_deg=None, angle_y_deg=None):
    x1, y1, x2, y2 = map(int, box[:4])
    box_color = get_color_by_confidence(confidence)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    if depth_value is not None:
        print(depth_value)
        label = f"{class_name} ({confidence:.2f}) | {depth_value:.2f}m"
    else:
        label = f"{class_name} ({confidence:.2f})"

    if angle_x_deg is not None:
        print("x degrees:" + str(round(angle_x_deg,1)))
        print("y degrees:" + str(round(angle_y_deg,1)))
        label = label + " | " + str(round(angle_x_deg,1)) + ", " + str(round(angle_x_deg,1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thick)[0]
    
    rect_x1 = x1
    rect_y1 = y1 - text_size[1] - 5
    rect_x2 = x1 + text_size[0] + 5
    rect_y2 = y1
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(
        image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thick
    )

def process_image(image_path, calibration_file=None):
    camera_params = CameraParams(calibration_file)
    
    # Load and resize image to match calibration size
    yolo_input = cv2.imread(image_path)
    original_size = yolo_input.shape[:2]
    yolo_input = cv2.resize(yolo_input, (512, 512))  # Match calibration size
    
    # Undistort at the calibrated resolution
    if camera_params.camera_matrix is not None:
        yolo_input = cv2.undistort(
            yolo_input,
            camera_params.camera_matrix,
            camera_params.dist_coeffs
        )
    
    start_yolo_time = time.time()
    results = yolo_model(yolo_input)
    yolo_inference_time = time.time() - start_yolo_time
    detected_boxes = []

    predictions = results.pandas().xyxy[0]
    
    for idx, detection in predictions.iterrows():
        conf = float(detection['confidence'])
        cls = int(detection['class'])
        
        if cls in CLASSES_TO_DETECT and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2), cls, conf))
            class_name = yolo_model.names[cls]
            draw_detection(yolo_input, [x1, y1, x2, y2], class_name, conf)

    if detected_boxes:
        midas_model, midas_transforms, device = load_midas_model()
        
        start_depth_time = time.time()
        img = cv2.cvtColor(yolo_input, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        input_batch = midas_transforms(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_np = prediction.cpu().numpy()
        depth_np = apply_depth_refinement(depth_np)
        depth_np = calculate_metric_depth(depth_np, camera_params)
        
        depth_inference_time = time.time() - start_depth_time

        angle_x_deg, angle_y_deg = calculate_object_angle((x1, y1, x2, y2), depth_np, camera_params)
        print(str(angle_x_deg) + " " + str(angle_y_deg))
        #draw_detection(yolo_input, [x1, y1, x2, y2], class_name, conf, depth_np[int((y1 + y2) / 2), int((x1 + x2) / 2)], angle_x_deg, angle_y_deg)

        for x1, y1, x2, y2, cls, conf in detected_boxes:
            box_depth = depth_np[y1:y2, x1:x2].mean()
            class_name = yolo_model.names[cls]
            box_with_depth = [x1, y1, x2, y2]
            draw_detection(yolo_input, box_with_depth, class_name, conf, box_depth, angle_x_deg, angle_y_deg)

    # Save and display results
    input_dir = os.path.dirname(image_path)
    file_name = "depth_write.png"
    output_path = os.path.join(input_dir, file_name)
    cv2.imwrite(output_path, yolo_input)

    print(f"YOLO inference time: {yolo_inference_time:.3f} seconds")
    if detected_boxes:
        print(f"Depth inference time: {depth_inference_time:.3f} seconds")

    cv2.imshow("Object Detection with Confidence and Depth", yolo_input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calibration utility')
    parser.add_argument('--calibrate', '-c', type=int, default=0, required=True, help='Calibrate image, 0 for no, 1 for yes (default is 0)')
    parser.add_argument('--depth_anything', '-d', type=int, default=0, required=True, help='Run depth anything v2 model, 0 for no, 1 for yes (default is 0)')
    args = parser.parse_args()
    if args.calibrate == 1 and args.depth_anything == 0:
        # First time setup: perform calibration
        calibrator = CameraCalibrator()
        calibrator.calibrate("./calibration-images")
    elif args.calibrate == 0 and args.depth_anything == 0:
        time.sleep(3)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        ret, frame = cap.read()
        time.sleep(1.5)
        cv2.imwrite("temp.jpg",frame)
        process_image("temp.jpg", "./camera_calibration.npy")
    elif args.calibrate == 0 and args.depth_anything == 1:
        load_depth_anything_model("temp.jpg")