from logging import config
import os
import argparse
import time
import random
import numpy as np
import cv2
import sys
from pathlib import Path
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
from PIL import Image, ImageDraw
from rim_detector import RimDetector
from picamera2 import Picamera2
import requests
import json

fps=10
frame_width=640
frame_height=640

label_map={
    0:"ball"
}

url = "https://api.kosko.rs/api/logs"
filename = 'scores.txt'

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "curl/7.68.0"
}
CONFIG_FILE = "/boot/overlays/cnf.txt"
SCORES_FILE = "scores.txt"

def load_config(path):
    with open(path, "r") as f:
        config = {}
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                config[k] = v

    if "I" not in config:
        raise RuntimeError("Missing device ID (I) in cnf.txt")

    if "P" not in config:
        config["P"] = "0"

    return config


def send_file_to_url(filename, url, device_id):
    with open(filename, 'r') as file:
        lines = file.readlines()

    p_value = next(
        (line.split(":")[1].strip() for line in lines if line.startswith("P:")),
        "0"
    )

    json_payload = {
        "id": device_id,
        "logs": [
            {
                "id": device_id,
                "points": int(p_value)
            }
        ]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=json_payload, headers=headers)

    if response.status_code == 200:
        with open(filename, "w") as file:
            file.write(f"I:{device_id}\nP:0\n")
            file.flush()
            os.fsync(file.fileno())
        return True

    print(f"Failed to send file: {response.status_code} {response.text}")
    return False


class BasketDetector:
    def __init__(self, rim_center, rim_radius,device_id ,strictness='medium'):
        self.device_id = device_id
        self.rim_center_x, self.rim_center_y = rim_center
        self.rim_radius_x, self.rim_radius_y = rim_radius
        self.ball_trackers = {}
        self.last_basket_time = time.time()
        
        # Adjusted settings to be more balanced
        self.strictness_settings = {
            'very_lenient': {
                'rim_zone_scale': 0.8,  # Slightly larger rim zone
                'min_confidence': 0.4,   # Lowered confidence threshold
                'area_change_threshold': 0.2,  # More lenient area change
                'min_positions': 2,      # Keep minimum tracking points
                'basket_interval': 1.0,
                'max_positions': 6,
                'min_velocity': 5,       # Lower velocity threshold
                'max_velocity': 150      # Add maximum velocity check
            }
        }
        
        self.set_strictness(strictness)
        self.basket_count = 0


        # Set current strictness
        self.set_strictness(strictness)
        self.ball_crossed_up = False
        self.basket_count = 0

    def set_strictness(self, level):
        if level not in self.strictness_settings:
            raise ValueError(f"Invalid strictness level. Choose from: {list(self.strictness_settings.keys())}")
        self.settings = self.strictness_settings[level]
        self.MAX_POSITIONS = self.settings['max_positions']
    def get_nearest_tracker(self, new_pos):
        """Find the nearest existing ball tracker or create a new one"""
        min_distance = float('inf')
        nearest_id = None
        
        for tracker_id, tracker_data in self.ball_trackers.items():
            if tracker_data['positions']:
                last_pos = tracker_data['positions'][-1][0]
                distance = ((new_pos[0] - last_pos[0])**2 + (new_pos[1] - last_pos[1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = tracker_id
        
        if min_distance < 140:
            return nearest_id
        
        new_id = max(self.ball_trackers.keys(), default=-1) + 1
        self.ball_trackers[new_id] = {
            'positions': [],
            'was_above_rim': False,
            'color': (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
        }
        return new_id



    def is_in_rim_zone(self, ball_pos):
        """Check if ball is within the rim's elliptical area"""
        dx = (ball_pos[0] - self.rim_center_x) / self.rim_radius_x
        dy = (ball_pos[1] - self.rim_center_y) / self.rim_radius_y
        return (dx * dx + dy * dy) <= self.settings['rim_zone_scale']

    def get_ball_area(self, box):
        """Calculate approximate ball area from bounding box"""
        width = abs(box[3] - box[1])
        height = abs(box[2] - box[0])
        return (width * height) * 2
    
    def calculate_velocity(self, positions):
        """Calculate velocity between last two positions"""
        if len(positions) < 2:
            return 0
        p1 = positions[-2][0]
        p2 = positions[-1][0]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (dx * dx + dy * dy) ** 0.5

    def is_valid_basket_movement(self, tracker):
        """Simplified validation of basket movement"""
        positions = tracker['positions']
        if len(positions) < self.settings['min_positions']:
            return False

        # Calculate basic movement metrics
        velocity = self.calculate_velocity(positions)
        if velocity < self.settings['min_velocity'] or velocity > self.settings['max_velocity']:
            return False

        # Check if the ball is moving consistently
        areas = [pos[1] for pos in tracker['positions']]
        area_change = (areas[-1] - areas[0]) / areas[0] if areas[0] > 0 else 0
        
        # The ball should show some significant change in size
        return area_change <= -self.settings['area_change_threshold']

    def calculate_trajectory_direction(self, positions):
        """Calculate overall trajectory direction and consistency"""
        if len(positions) < 3:
            return 0, 0
            
        directions = []
        for i in range(len(positions) - 1):
            dx = positions[i+1][0][0] - positions[i][0][0]
            dy = positions[i+1][0][1] - positions[i][0][1]
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                directions.append(angle)
        
        if not directions:
            return 0, 0
            
        # Calculate direction consistency
        consistency = np.std(directions) if len(directions) > 1 else float('inf')
        avg_direction = np.mean(directions)
        return avg_direction, consistency

    def is_valid_shot_trajectory(self, tracker):
        """Verify if the ball's trajectory matches expected shot characteristics"""
        positions = tracker['positions']
        if len(positions) < self.settings['min_trajectory_length']:
            return False

        # Calculate trajectory characteristics
        direction, consistency = self.calculate_trajectory_direction(positions)
        
        # Check if trajectory is consistent enough
        if consistency > np.radians(self.settings['rim_entry_angle']):
            return False
            
        # Calculate vertical movement
        total_vertical = positions[-1][0][1] - positions[0][0][1]
        
        # For top-down view, we expect mostly horizontal movement
        horizontal_movement = abs(positions[-1][0][0] - positions[0][0][0])
        if horizontal_movement < self.rim_radius_x * 0.5:  # Minimum expected horizontal movement
            return False

        # Check area changes for consistent ball movement
        areas = [pos[1] for pos in positions]
        area_changes = np.diff(areas)
        if np.std(area_changes) > np.mean(areas) * 0.5:  # Check for smooth area changes
            return False

        return True

    def detect_basket(self, frame, results, current_count):
        current_time = time.time()
        active_trackers = set()
        
        for i in range(int(results['num_detections'][0])):
            class_num = int(results['detection_classes'][0][i])
            confidence = float(results['detection_scores'][0][i])
            
            if class_num == 0 and confidence >= self.settings['min_confidence']:
                box = results['detection_boxes'][0][i]
                ball_area = self.get_ball_area(box)
                if(ball_area<0.05 or ball_area>0.2 ):
                    continue
                #print(ball_area)
                ball_center = (
                    int((box[1] + box[3]) * frame.shape[1] / 2),
                    int((box[0] + box[2]) * frame.shape[0] / 2)
                )
                
                tracker_id = self.get_nearest_tracker(ball_center)
                active_trackers.add(tracker_id)
                tracker = self.ball_trackers[tracker_id]
                
                # Update positions
                tracker['positions'].append((ball_center, ball_area))
                if len(tracker['positions']) > self.settings['max_positions']:
                    tracker['positions'].pop(0)
                
                # Track whether ball has been above rim
                if not tracker.get('was_above_rim', False):
                    # Check if ball is above rim in top-down view (closer to center)
                    dx = ball_center[0] - self.rim_center_x
                    dy = ball_center[1] - self.rim_center_y
                    dist_to_center = (dx * dx + dy * dy) ** 0.5
                    if dist_to_center < self.rim_radius_x * 0.6:  # If ball is near center
                        tracker['was_above_rim'] = True
                
                # Simplified basket detection logic
                if len(tracker['positions']) >= self.settings['min_positions']:
                    in_rim = self.is_in_rim_zone(ball_center)
                    valid_movement = self.is_valid_basket_movement(tracker)
                    
                    # Ball must have been above rim at some point
                    if (in_rim and 
                        valid_movement and 
                        tracker.get('was_above_rim', False) and
                        current_time - self.last_basket_time > self.settings['basket_interval']):
                        
                        current_count += 1
                        self.last_basket_time = current_time
                        tracker['was_above_rim'] = False  # Reset for next potential basket
                        
                        with open(filename, "w") as file:
                            formatted_string = f"I:{self.device_id}\nP:{current_count}\n"
                            file.write(formatted_string)
                            file.flush()
                            os.fsync(file.fileno())

                        cv2.putText(frame, "BASKET!", 
                                  (int(self.rim_center_x) - 50, int(self.rim_center_y) - 20),
                                  cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 4)
                
                # Visualization
                cv2.circle(frame, ball_center, 5, tracker.get('color', (0,255,0)), -1)
                if len(tracker['positions']) >= 2:
                    points = np.array([pos[0] for pos in tracker['positions']], dtype=np.int32)
                    cv2.polylines(frame, [points.reshape((-1, 1, 2))], False, tracker.get('color', (0,255,0)), 2)
        
        # Clean up inactive trackers
        inactive = set(self.ball_trackers.keys()) - active_trackers
        for tracker_id in inactive:
            del self.ball_trackers[tracker_id]
        
        return current_count, frame
    
def initialize_detector(avg_center_x, avg_center_y, avg_max_radius, avg_min_radius,device_id ,strictness='lenient'):
    """Initialize the detector with rim parameters and strictness level"""
    rim_center = (avg_center_x, avg_center_y)
    rim_radius = (avg_max_radius/2, avg_min_radius/2)
    return BasketDetector(rim_center, rim_radius, device_id, strictness)

def calculate_stats(times):
    """Calculate statistics for inference times"""
    if not times:
        return 0, 0, 0, 0

    avg_time = np.mean(times)
    std_dev = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    return avg_time, std_dev, min_time, max_time

def setup_logging():
    """Setup logging to stdout instead of file"""
    # Disable file logging by setting environment variable
    os.environ['HAILO_LOG_TO_CONSOLE'] = '1'
    os.environ['HAILO_LOG_FILE'] = ''

def validate_video_format(text):
    """Validate video file extension"""
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return text.lower().endswith(valid_extensions)

def get_video_writer(output_path, fps, width, height):
    """Create video writer with proper codec based on OS and file extension"""
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Fallback to XVID

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Warning: Video writer failed to open, using fallback codec.")
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if sys.platform == "darwin":  # macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:  # Linux/Windows
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # Try fallback codec
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return writer

def generate_new_color(color_dict):
    if len(color_dict.keys()) == 0:
        return [0, 0, 0]
    if len(color_dict.keys()) == 1:
        return [0, 255, 0]
    if len(color_dict.keys()) == 2:
        return [0, 0, 255]

    new_color = [random.randint(0, 255) for _ in range(3)]
    while new_color in color_dict.values():
        new_color = [random.randint(0, 255) for _ in range(3)]
    return new_color

def draw_boxes(frame, results, used_colors, label_map):
    for i in range(int(results['num_detections'][0])):
        # Get detection details
        class_num = int(results['detection_classes'][0][i])
        confidence = float(results['detection_scores'][0][i])

        # Skip low confidence detections if needed
        if confidence < 0.3:  # You can adjust this threshold
            continue

        # Generate color for new classes
        if class_num not in used_colors:
            used_colors[class_num] = generate_new_color(used_colors)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Calculate bounding box coordinates
        start_point = (
            round(results['detection_boxes'][0][i][1] * width),
            round(results['detection_boxes'][0][i][0] * height)
        )
        end_point = (
            round(results['detection_boxes'][0][i][3] * width),
            round(results['detection_boxes'][0][i][2] * height)
        )
        # Draw bounding box
        color = tuple(used_colors[class_num])
        cv2.rectangle(frame, start_point, end_point, color, 2)

        # Prepare label text with class name and confidence
        label = label_map.get(class_num, f"Class {class_num}")
        label_text = f"{label}: {confidence:.2f}"

        # Add label above the bounding box
        cv2.putText(frame, label_text, (start_point[0], start_point[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def extract_detections(inputs, boxes, scores, classes, num_detections):
    for i, detection in enumerate(inputs):
        if len(detection) == 0:
            continue
        for j in range(len(detection)):
            bbox = np.array(detection)[j][:4]
            score = np.array(detection)[j][4]
            boxes.append(bbox)
            scores.append(score)
            classes.append(i)
            num_detections = num_detections + 1
    return {
        'detection_boxes': [boxes],
        'detection_classes': [classes],
        'detection_scores': [scores],
        'num_detections': [num_detections]
    }

def post_nms_infer(raw_detections, input_name):
    boxes = []
    scores = []
    classes = []
    num_detections = 0

    detections = extract_detections(
        raw_detections[input_name][0],
        boxes,
        scores,
        classes,
        num_detections
    )

    return detections

def parse_args():
    parser = argparse.ArgumentParser(
        description='Running Hailo inference on video using Hailo API')
    parser.add_argument('hef', help="HEF file path")
    parser.add_argument(
        'input_video',
        help="Path to input video file (mp4, avi, mov, or mkv)")
    parser.add_argument(
        'output_video',
        help="Path to output video file (mp4, avi, mov, or mkv)")
    parser.add_argument(
        '--class-num',
        help="The number of classes the model is trained on. Defaults to 80",
        default=80)
    parser.add_argument(
        '--labels',
        default='',
        help="The path to the labels txt file. Should be in a form of NUM : LABEL")
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='detection threshold before a box is accepted')
    parser.add_argument('-u', '--iou', type=float, default=0.5,
                        help='intersect over union during nms to merge boxes')
    return parser.parse_args()




# [BasketDetector class remains unchanged]
# [Other helper functions remain unchanged]


def main():
    config = load_config(CONFIG_FILE)
    DEVICE_ID = config["I"]
    try:
        args = parse_args()

        if not os.path.exists(args.hef):
            raise FileNotFoundError(f"HEF file not found: {args.hef}")

        # Initialize PiCamera2 instead of video file
        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"}  # Force 3-channel output
)
#1640 x 1232
        picam2.configure(camera_config)
        picam2.start()

        # Get frame dimensions from camera configuration
        frame_width = 640
        frame_height = 640
        fps = 30  # Assuming 30 fps, adjust as needed for your Pi camera setup

        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

        devices = Device.scan()
        if not devices:
            raise RuntimeError("No Hailo devices found")

        hef = HEF(args.hef)
        outputs = hef.get_output_vstream_infos()
        rim_detector = RimDetector(debug_mode=False)
        basket_count = 0
        rim_locked = False
        global total_frames
        total_frames = 0


        # Variable to track when to perform rim detection
        frames_since_detection = 0

        with VDevice(device_ids=devices) as target:
            configure_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()
            used_colors = {}

            input_vstreams_params = InputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.UINT8)
            output_vstreams_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32)

            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                input_vstream_info = hef.get_input_vstream_infos()[0]
                infer_pipeline.set_nms_iou_threshold(1.0)
                infer_pipeline.set_nms_score_threshold(0.5)

                with network_group.activate(network_group_params):
                    while True:
                        # Capture frame from PiCamera2
                        frame = picam2.capture_array()
                        
                        # Ensure frame is resized to expected dimensions
                        frame = cv2.resize(frame, (640, 640))
                        total_frames += 1
                        frames_since_detection += 1

                        # Only process rim detection every 8000 frames or if rim is not yet locked
                        should_detect_rim = frames_since_detection >= 1000 or not rim_locked
                        
                        if should_detect_rim:
                            frames_since_detection = 0  # Reset counter
                            frame_rgb=frame
                            # Convert to RGB for model input
                            input_data = {input_vstream_info.name: np.expand_dims(frame_rgb, axis=0).astype(np.uint8)}

                            raw_detections = infer_pipeline.infer(input_data)
                            results = post_nms_infer(raw_detections, outputs[0].name)

                            frame = draw_boxes(frame, results, used_colors, label_map)
                            ellipse = rim_detector.detect_rim(frame)

                            # Debugging statement for rim detection

                            if ellipse is not None:
                                rim_detector.draw_rim(frame, ellipse)
                                if not rim_locked:
                                    (center_x, center_y), (max_radius, min_radius), _ = ellipse
                                    rim_locked = True
                                    avg_center_x = center_x
                                    avg_center_y = center_y
                                    avg_min_radius = min_radius
                                    avg_max_radius = max_radius

                            if rim_locked and 'detector' not in locals():
                                print("Initializing BasketDetector...")
                                detector = initialize_detector(avg_center_x, avg_center_y, avg_max_radius, avg_min_radius,DEVICE_ID ,'very_lenient')
                        
                        # Always check for baskets if rim is locked (even between detection intervals)
                        if rim_locked:
                            rim_detector.draw_rim(frame, ellipse)
                            # We need to process object detection for basket detection
                            if frames_since_detection != 0:  # Only if we didn't just do detection above
                                frame_rgb=frame
                                input_data = {input_vstream_info.name: np.expand_dims(frame_rgb, axis=0).astype(np.uint8)}
                                raw_detections = infer_pipeline.infer(input_data)
                                results = post_nms_infer(raw_detections, outputs[0].name)
                                frame = draw_boxes(frame, results, used_colors, label_map)
                                basket_count, frame = detector.detect_basket(frame, results, basket_count)
                        if(total_frames % 10000 ==0):
                            print(f"Current count: {basket_count} , Frames {total_frames}")
                            

                        # Overlay text
                        
                        
                        #video_writer.write(frame)
                        
                        if total_frames % 90000 == 0:
                            if(send_file_to_url(filename, url,DEVICE_ID)):
                                basket_count=0

                            
                            
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

    finally:
        if 'video_writer' in locals():
            video_writer.release()
        if 'picam2' in locals():
            picam2.stop()
        cv2.destroyAllWindows()
    global frame_total
    frame_total = total_frames

if __name__ == '__main__':
    main()
