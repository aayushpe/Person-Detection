import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Path to the video file
video_path = "people.mp4" 

# Bounding zone 
bounding_zone = np.array([[4, 156], [131, 97], [149, 219], [204, 304], [215, 296], [235, 331], [357, 330], [365, 21], [465, 124], [640, 411], [640, 640], [0, 640]])

# Load YOLOv8 NCNN model -> finetuned for person detection through CCTV footage
model = YOLO("models/person_ncnn_model")

# The Pytorch Model could also be loaded instead (slower but more accurate)
# model = YOLO("models/person.pt")

# The Base YOLO based of the COCO dataset (80 different classes)
# model = YOLO("models/yolov8n.pt")

# Function to create a tracking zone
def create_zone(zone_coords):
    return sv.PolygonZone(polygon=zone_coords)

# Function to draw the zone
def draw_zone(frame, zone):
    if zone and len(zone.polygon) > 0:
        points = zone.polygon
        cv2.polylines(frame, [points], isClosed=True, color=(251, 81, 163), thickness=2)
    return frame

# Process frame for detection and tracking
def process_frame(frame, model, tracker, trace_annotator, zone):
    if frame is None:
        return None, 0

    # Resize frame for better inference performance
    frame_resized = cv2.resize(frame, (640, 640))

    # Perform detection using YOLO
    results = model(frame_resized, conf=0.25, iou=0.4)
    detections = sv.Detections.from_ultralytics(results[0])

    # Update tracker with detections
    tracked_objects = tracker.update_with_detections(detections)

    # Apply zone mask
    if zone:
        mask = zone.trigger(detections=tracked_objects)
        tracked_objects = tracked_objects[mask]

    # Annotate the frame
    bbox_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    annotated_frame = bbox_annotator.annotate(scene=frame_resized, detections=tracked_objects)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_objects)

    # Add traces to annotated frame
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_objects)

    # Count objects
    object_count = len(tracked_objects)

    return annotated_frame, object_count

# Display counts and warnings on the frame
def display_counts_and_warnings(frame, count):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Display the total count in black
    text = f"Count: {count}"
    text_color = (0, 0, 0)  # Black text
    cv2.putText(frame, text, (10, 30), font, 1, text_color, 2, cv2.LINE_AA)

    # Display warning if count exceeds 8
    if count > 5:
        warning_text = "WARNING: Too many people!"
        warning_color = (0, 0, 255)  # Red text

        # Calculate text size for centering
        text_size, _ = cv2.getTextSize(warning_text, font, 1.5, 3)
        text_width, text_height = text_size
        center_x = (640 - text_width) // 2
        center_y = (640 + text_height) // 2

        # Display the warning
        cv2.putText(frame, warning_text, (center_x, center_y), font, 1.5, warning_color, 3, cv2.LINE_AA)

    return frame

# Main function
def main():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    tracker = sv.ByteTrack()
    trace_annotator = sv.TraceAnnotator()
    zone = create_zone(bounding_zone)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        annotated_frame, object_count = process_frame(frame, model, tracker, trace_annotator, zone)

        # Draw the zone
        if annotated_frame is not None:
            annotated_frame = draw_zone(annotated_frame, zone)
            annotated_frame = display_counts_and_warnings(annotated_frame, object_count)

        # Display the frame
        cv2.imshow("YOLOv8 + ByteTrack", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()