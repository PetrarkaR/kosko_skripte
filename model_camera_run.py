import cv2
import numpy as np
import onnxruntime as ort
import time
from flask import Flask, Response
import threading

app = Flask(__name__)

class YOLOBallDetector:
    def __init__(self, onnx_path):
        # Use CPU execution provider for Raspberry Pi
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Adjust for sports ball detection
        self.ball_class_ids = [32]  # COCO dataset sports ball class
        self.custom_labels = {32: "sports_ball"}
        
        # Shared frame for streaming
        self.current_frame = None
        self.lock = threading.Lock()

    def detect_objects(self, image, confidence_threshold=0.3, nms_threshold=0.5):
        # Get original image dimensions
        height, width, _ = image.shape

        # Resize image for inference (optimized for performance)
        input_size = (416, 416)  # Smaller size for faster processing on Raspberry Pi
        resized_image = cv2.resize(image, input_size)
        
        # Preprocessing
        blob = resized_image.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # Convert to CxHxW format
        blob = np.expand_dims(blob, axis=0)  # Add batch dimension

        # Inference
        inputs = {self.session.get_inputs()[0].name: blob}
        outputs = self.session.run(None, inputs)

        # Process the outputs
        boxes, confidences, class_ids = self._process_detections(outputs, width, height, confidence_threshold)

        # Apply Non-Maximum Suppression
        try:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        except:
            # Handle potential errors with NMS
            indexes = []

        # Prepare results
        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.custom_labels.get(class_ids[i], f"Class_{class_ids[i]}")
                confidence = confidences[i]
                results.append({
                    'box': [x, y, w, h],
                    'class': label,
                    'confidence': confidence
                })

        return results

    def _process_detections(self, outputs, width, height, confidence_threshold):
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output[0]:
                scores = detection[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold and class_id in self.ball_class_ids:
                    center_x, center_y, w, h = detection[:4]
                    center_x, center_y = int(center_x * width), int(center_y * height)
                    w, h = int(w * width), int(h * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_detections(self, image, detections):
        output = image.copy()
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection['class']
            confidence = detection['confidence']

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(output, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output

    def start_detection(self):
        # Open camera capture
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally (optional)
            frame = cv2.flip(frame, 1)

            # Light preprocessing
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # Detect objects
            detections = self.detect_objects(frame)

            # Draw detections
            output_frame = self.draw_detections(frame, detections)

            # Update shared frame with thread safety
            with self.lock:
                self.current_frame = output_frame

            # Performance tracking
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

    def get_frame(self):
        # Retrieve the current frame with thread safety
        with self.lock:
            if self.current_frame is None:
                return None
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            return buffer.tobytes()

# Global detector instance
detector = YOLOBallDetector("/home/pi/ball_detection/yolov5l.onnx")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = detector.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Ball Detection Stream</title>
    </head>
    <body>
        <h1>Ball Detection Video Stream</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

def main():
    # Start detection in a separate thread
    detection_thread = threading.Thread(target=detector.start_detection)
    detection_thread.daemon = True
    detection_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()