#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "httplib.h"
#include "rknn_api.h"
#include "json.hpp"

using json = nlohmann::json;

#define DEPTH_ANYTHING_MODEL_PATH "../models/depth-anything.rknn"
#define YOLO_MODEL_PATH "../models/yolov5.rknn"
#define DEFAULT_IMG_PATH "../test.jpg"
#define DEFAULT_CAMERA_DEVICE 0

cv::Mat captured_image;

// Structure to represent a detected object with depth information
struct DetectedObject {
    std::string objectClass;
    float objectDistance;
    float objectLateralAngle;
    float objectVerticalAngle;
    cv::Rect bbox;
    float confidence;
};

// Structure to hold combined model outputs
struct ModelResults {
    std::vector<DetectedObject> objects;
};

// YOLO classes (modify according to your model)
const std::vector<std::string> YOLO_CLASSES = {
    "coral", "algae", "reef", "person"
};

cv::Mat takeSnapshotFromSelectedCamera(const int camera_id) {
    cv::VideoCapture camera(camera_id);
    if (!camera.isOpened()) {
        std::cerr << "Could not open camera " << camera_id << std::endl;
        return cv::Mat();
    }
    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) {
        std::cerr << "Could not grab frame from camera " << camera_id << std::endl;
        return cv::Mat();
    }
    return frame; 
    //return img;
}

// Function to load an image and preprocess it for the model
cv::Mat loadAndPreprocessImage(const cv::Mat& frame, int target_width, int target_height) {
    /*if(frame.empty() && !image_path.empty()){
        cv::Mat orig_img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (orig_img.empty()) {
            fprintf(stderr, "Open image failed: %s\n", image_path.c_str());
            return cv::Mat();
        }

        // Resize image to match model input dimensions
        cv::Mat img;
        cv::resize(orig_img, img, cv::Size(target_width, target_height));
        
        // Convert to RGB (OpenCV loads as BGR)
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        captured_image = img;
        return img;
    } else {*/
        cv::Mat img;
        cv::resize(frame, img, cv::Size(target_width, target_height));
        
        // Convert to RGB (OpenCV loads as BGR)
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        
        captured_image = img;
        return img;
    //}    
}

// Function to run YOLOv5
std::vector<cv::Rect> runYolo(std::vector<int>& class_ids, std::vector<float>& confidences, int camera_id) {
    int ret;
    rknn_context ctx;
    std::vector<cv::Rect> bboxes;

    // 1. Load RKNN model
    ret = rknn_init(&ctx, (void*)(YOLO_MODEL_PATH), 0, 0, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return bboxes;
    }

    // 2. Get model input/output info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
        rknn_destroy(ctx);
        return bboxes;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            return bboxes;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            return bboxes;
        }
    }

    // 3. Load and preprocess the image
    int input_width = input_attrs[0].dims[2];
    int input_height = input_attrs[0].dims[1];
    cv::Mat img = loadAndPreprocessImage(takeSnapshotFromSelectedCamera(camera_id), input_width, input_height);
    if (img.empty()) {
        rknn_destroy(ctx);
        return bboxes;
    }

    // 4. Prepare input tensor
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;
    inputs[0].size = img.total() * img.channels();

    // 5. Run inference
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        fprintf(stderr, "rknn_input_set error ret=%d\n", ret);
        rknn_destroy(ctx);
        return bboxes;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_run error ret=%d\n", ret);
        rknn_destroy(ctx);
        return bboxes;
    }

    // 6. Get output data
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_outputs_get error ret=%d\n", ret);
        rknn_destroy(ctx);
        return bboxes;
    }

    // 7. Process YOLO output
    // Note: This is a simplified implementation. Adjust according to your YOLO model's output format
    float* output_data = (float*)outputs[0].buf;
    
    // Assuming output is in format [batch, num_boxes, 5+num_classes]
    // where 5 = [x, y, w, h, conf]
    int num_boxes = output_attrs[0].dims[1];
    int box_info_size = output_attrs[0].dims[2];
    float conf_threshold = 0.4;  // Adjust as needed
    
    // Original image dimensions (for scaling bounding boxes)
    cv::Mat orig_img = captured_image;
    float scale_x = float(orig_img.cols) / input_width;
    float scale_y = float(orig_img.rows) / input_height;
    
    for (int i = 0; i < num_boxes; i++) {
        float* box_data = output_data + i * box_info_size;
        float box_conf = box_data[4];
        
        if (box_conf > conf_threshold) {
            // Find class with max probability
            int max_class_id = 0;
            float max_class_prob = 0;
            for (int j = 5; j < box_info_size; j++) {
                if (box_data[j] > max_class_prob) {
                    max_class_prob = box_data[j];
                    max_class_id = j - 5;
                }
            }
            
            // Calculate bounding box coordinates
            float x = box_data[0] * scale_x;
            float y = box_data[1] * scale_y;
            float w = box_data[2] * scale_x;
            float h = box_data[3] * scale_y;
            
            // Convert to top-left, width, height format
            int x1 = std::max(0, int(x - w/2));
            int y1 = std::max(0, int(y - h/2));
            int width = int(w);
            int height = int(h);
            
            // Store bounding box, class ID, and confidence
            bboxes.push_back(cv::Rect(x1, y1, width, height));
            class_ids.push_back(max_class_id);
            confidences.push_back(box_conf);
        }
    }
    
    // 8. Release resources
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_outputs_release(ctx, 1, &outputs[i]);
    }
    rknn_destroy(ctx);
    
    return bboxes;
}

// Function to run Depth Anything
std::vector<std::vector<float>> runDepthAnything(int camera_id) {
    int ret;
    rknn_context ctx;
    std::vector<std::vector<float>> depth_map;

    // 1. Load RKNN model
    ret = rknn_init(&ctx, (void*)(DEPTH_ANYTHING_MODEL_PATH), 0, 0, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return depth_map;
    }

    // 2. Get model input/output info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
        rknn_destroy(ctx);
        return depth_map;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            return depth_map;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            return depth_map;
        }
    }

    // 3. Load and preprocess the image
    int input_width = input_attrs[0].dims[2];
    int input_height = input_attrs[0].dims[1];
    cv::Mat img = loadAndPreprocessImage(takeSnapshotFromSelectedCamera(camera_id), input_width, input_height);
    if (img.empty()) {
        rknn_destroy(ctx);
        return depth_map;
    }

    // 4. Prepare input tensor
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;
    inputs[0].size = img.total() * img.channels();

    // 5. Run inference
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        fprintf(stderr, "rknn_input_set error ret=%d\n", ret);
        rknn_destroy(ctx);
        return depth_map;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_run error ret=%d\n", ret);
        rknn_destroy(ctx);
        return depth_map;
    }

    // 6. Get output data
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_outputs_get error ret=%d\n", ret);
        rknn_destroy(ctx);
        return depth_map;
    }

    // 7. Process depth map output
    // Assuming output 0 is the depth map
    int output_index = 0;  // Adjust if depth output is at different index
    int height = output_attrs[output_index].dims[1];
    int width = output_attrs[output_index].dims[2];
    
    float* output_data = (float*)outputs[output_index].buf;
    
    // Create depth map with proper dimensions
    depth_map.resize(height);
    for (int h = 0; h < height; h++) {
        depth_map[h].resize(width);
        for (int w = 0; w < width; w++) {
            // Get value from flattened output array
            int index = h * width + w;
            depth_map[h][w] = output_data[index];
        }
    }
    
    // 8. Release resources
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_outputs_release(ctx, 1, &outputs[i]);
    }
    rknn_destroy(ctx);

    return depth_map;
}

// Calculate angles relative to center of frame
void calculateAngles(const cv::Rect& bbox, const cv::Mat& orig_img, float& lateral_angle, float& vertical_angle) {
    // Calculate center point of bounding box
    float center_x = bbox.x + bbox.width / 2.0f;
    float center_y = bbox.y + bbox.height / 2.0f;
    
    // Calculate image center
    float img_center_x = orig_img.cols / 2.0f;
    float img_center_y = orig_img.rows / 2.0f;
    
    // Calculate offsets from center (normalized to [-1, 1])
    float offset_x = (center_x - img_center_x) / img_center_x; 
    float offset_y = (center_y - img_center_y) / img_center_y;
    
    // Simple conversion to angles (assuming 90-degree FOV for camera)
    // This is a simplified calculation - adjust based on your camera's FOV
    float field_of_view_x = 90.0f; // horizontal FOV in degrees
    float field_of_view_y = 70.0f; // vertical FOV in degrees
    
    lateral_angle = offset_x * (field_of_view_x / 2.0f);
    vertical_angle = offset_y * (field_of_view_y / 2.0f);
}

// Convert depth values to real-world distance
float calculateDistance(float depth_value) {
    // This is a simple conversion example - you'll need to calibrate based on your depth model
    // Depth values are often relative, not absolute meters
    // You might need a scaling factor based on empirical testing
    float scaling_factor = 10.0f; // Example value, needs calibration
    return depth_value * scaling_factor;
}

// Process with both models and return combined results
ModelResults processWithBothModels(int camera_id) {
    ModelResults results;
    
    // Run object detection
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> bboxes = runYolo(class_ids, confidences, camera_id);
    
    // Run depth estimation
    std::vector<std::vector<float>> depth_map = runDepthAnything(camera_id);
    
    // If either model failed, return empty results
    if (bboxes.empty() || depth_map.empty()) {
        return results;
    }
    
    // Load original image for calculating angles
    cv::Mat orig_img = captured_image;
    if (orig_img.empty()) {
        return results;
    }
    
    // Calculate scale factors to map depth map to original image
    float scale_x = float(depth_map[0].size()) / orig_img.cols;
    float scale_y = float(depth_map.size()) / orig_img.rows;
    
    // Process each detected object
    for (size_t i = 0; i < bboxes.size(); i++) {
        DetectedObject obj;
        obj.objectClass = (class_ids[i] < YOLO_CLASSES.size()) ? YOLO_CLASSES[class_ids[i]] : "unknown";
        obj.bbox = bboxes[i];
        obj.confidence = confidences[i];
        
        // Calculate center of bounding box
        int center_x = bboxes[i].x + bboxes[i].width / 2;
        int center_y = bboxes[i].y + bboxes[i].height / 2;
        
        // Map to depth map coordinates
        int depth_x = int(center_x * scale_x);
        int depth_y = int(center_y * scale_y);
        
        // Ensure coordinates are within bounds
        depth_x = std::max(0, std::min(depth_x, int(depth_map[0].size()) - 1));
        depth_y = std::max(0, std::min(depth_y, int(depth_map.size()) - 1));
        
        // Get depth value and convert to distance
        float depth_value = depth_map[depth_y][depth_x];
        obj.objectDistance = calculateDistance(depth_value);
        
        // Calculate angles
        calculateAngles(bboxes[i], orig_img, obj.objectLateralAngle, obj.objectVerticalAngle);
        
        results.objects.push_back(obj);
    }
    
    // Sort objects by distance (closest first)
    std::sort(results.objects.begin(), results.objects.end(),
              [](const DetectedObject& a, const DetectedObject& b) {
                  return a.objectDistance < b.objectDistance;
              });
    
    return results;
}

// Convert distance to human-readable string
std::string formatDistance(float meters) {
    char buffer[32];
    sprintf(buffer, "%.1f meters", meters);
    return std::string(buffer);
}

int main() {
    //crow::SimpleApp app;

    /*CROW_ROUTE(app, "/")([](){
        return "Object Detection and Depth Estimation API";
    //});

    CROW_ROUTE(app, "/get_all_objects")([](const crow::request& req){
        // Get image path from query parameter or use default
        std::string image_path = DEFAULT_IMG_PATH;
        if (req.url_params.get("image") != nullptr) {
            image_path = req.url_params.get("image");
        }
        
        // Process image with both models
        ModelResults results = processWithBothModels(image_path);
        
        // Convert to JSON
        json response;
        response["Objects"] = json::array();
        
        for (const auto& obj : results.objects) {
            json object_data;
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
        
        return crow::response(response.dump(4));
    });

    CROW_ROUTE(app, "/get_closest_object")([](const crow::request& req){
        // Get image path from query parameter or use default
        std::string image_path = DEFAULT_IMG_PATH;
        if (req.url_params.get("image") != nullptr) {
            image_path = req.url_params.get("image");
        }
        
        // Process image with both models
        ModelResults results = processWithBothModels(image_path);
        
        // Convert to JSON (only closest object)
        json response;
        response["Objects"] = json::array();
        
        if (!results.objects.empty()) {
            json object_data;
            const auto& obj = results.objects[0]; // Get the closest object (already sorted)
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
        
        return crow::response(response.dump(4));
    });

    app.port(8008).multithreaded().run();
    */
    httplib::Server svr;
    
    svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("Object Detection and Depth Estimation API", "text/plain");
    });
    
    svr.Get("/get_closest_object", [](const httplib::Request &req, httplib::Response &res) {
        std::string cameraId_as_string = std::to_string(DEFAULT_CAMERA_DEVICE);
        if (req.has_param("camera")) {
            cameraId_as_string = req.get_param_value("camera");
        }

        int cameraId_as_int = std::stoi(cameraId_as_string);
        // Process image with both models
        ModelResults results = processWithBothModels(cameraId_as_int);
        
        // Convert to JSON (only closest object)
        json response;
        response["Objects"] = json::array();
        
        if (!results.objects.empty()) {
            json object_data;
            const auto& obj = results.objects[0]; // Get the closest object (already sorted)
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
        res.set_content(response.dump(4), "application/json");
    });

    svr.Get("/get_all_objects", [](const httplib::Request &req, httplib::Response &res){
        // Get image path from query parameter or use default
        std::string cameraId_as_string = std::to_string(DEFAULT_CAMERA_DEVICE);
        if (req.has_param("camera")) {
            cameraId_as_string = req.get_param_value("camera");
        }

        int cameraId_as_int = std::stoi(cameraId_as_string);
        // Process image with both models
        ModelResults results = processWithBothModels(cameraId_as_int);
        
        // Convert to JSON
        json response;
        response["Objects"] = json::array();
        
        for (const auto& obj : results.objects) {
            json object_data;
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
    });

    svr.listen("0.0.0.0", 8008);
    
    return 0;
}