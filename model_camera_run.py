import cv2
import numpy as np
import onnxruntime as ort
import time

class YOLOBallDetector:
    def __init__(self, onnx_path):
        # Use CPU execution provider for Raspberry Pi
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Adjust for sports ball detection
        self.ball_class_ids = [32]  # COCO dataset sports ball class
        self.custom_labels = {32: "sports_ball"}

    def detect_objects(self, image, confidence_threshold=0.3, nms_threshold=0.5):
        # Get original image dimensions
        height, width, _ = image.shape

        # Resize image for inference (optimized for performance)
        input_size = (320, 320)  # Smaller size for faster processing on Raspberry Pi
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


def main():
    # Path to ONNX model (adjust for Raspberry Pi file system)
    onnx_path = "/home/pi/ball_detection/yolov5l.onnx"

    # Create detector
    detector = YOLOBallDetector(onnx_path)

    # Open camera capture
    try:
        # Try different camera indices if needed
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution (adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally (optional)
            frame = cv2.flip(frame, 1)

            # Light preprocessing to reduce computational load
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # Detect objects
            detections = detector.detect_objects(frame)

            # Draw detections
            output_frame = detector.draw_detections(frame, detections)

            # Display frame
            cv2.imshow('Ball Detection', output_frame)

            # Performance tracking
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()