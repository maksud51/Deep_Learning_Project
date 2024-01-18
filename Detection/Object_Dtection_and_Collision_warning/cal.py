import cv2
import argparse
from ultralytics import YOLO
import parameter



parser = argparse.ArgumentParser(description="Object Detection on Video")
parser.add_argument("--path", required=True, help="Path to the video file")
args = parser.parse_args()
yolo_model = YOLO(parameter.model_name)
class_names = yolo_model.names



video_capture = cv2.VideoCapture(args.path)
if not video_capture.isOpened():
    print(f"Error opening video file: {args.path}")
    exit(1)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = yolo_model.predict(
        source=frame,
        imgsz=parameter.image_size,
        conf=parameter.confidence,
        save=False,
        classes=parameter.selected_class
    )

    detection_boxes = results[0].boxes.data.tolist()
    for box in detection_boxes:
        x1, y1, x2, y2, confidence, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])
        normalized_width = (x2 - x1) / width
        distance = parameter.distance_calculator(normalized_width)
        label = f'**warning**' if distance <= parameter.distance_parameter else f'{class_names[int(cls)]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if distance <= parameter.distance_parameter else (255, 125, 120), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if distance <= parameter.distance_parameter else (255, 125, 120), 1)

    cv2.imshow("Collision Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
