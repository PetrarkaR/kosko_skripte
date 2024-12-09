import cv2
import numpy as np
import onnxruntime as ort

class YOLOBallDetector:
    def __init__(self, onnx_path):
        # Load the ONNX model
        self.session = ort.InferenceSession(onnx_path)
        self.ball_class_ids = [32]  # Default 'sports_ball' class in COCO
        self.custom_labels = {32: "sports_ball"}  # Add custom labels if needed

    def detect_objects(self, image, confidence_threshold=0.3, nms_threshold=0.5):
        height, width, _ = image.shape

        # Preprocessing: Resize image to match model input
        resized_image = cv2.resize(image, (640, 640))
        blob = resized_image.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # Convert to CxHxW format
        blob = np.expand_dims(blob, axis=0)  # Add batch dimension

        # Inference
        inputs = {self.session.get_inputs()[0].name: blob}
        outputs = self.session.run(None, inputs)

        # Process the outputs
        boxes, confidences, class_ids = self._process_detections(outputs, width, height, confidence_threshold)

        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

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
    # Path to YOLOv5 Nano ONNX model
    onnx_path = r"C:\Users\swagg\OneDrive\Desktop\kosko\yolov5\yolov5l.onnx"

    # Create detector
    detector = YOLOBallDetector(onnx_path)

    # Open video capture
    video_path = r"C:\Users\swagg\OneDrive\Desktop\kosko\Video snimci\2024-03-01-113629.webm"
    cap = cv2.VideoCapture(video_path)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_detection_balls_640.mp4', fourcc, fps, (width, height))

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

        # Detect objects
        detections = detector.detect_objects(frame)

        # Draw detections
        output_frame = detector.draw_detections(frame, detections)

        # Write and show frame
        out.write(output_frame)
        cv2.imshow('Ball Detection', output_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
