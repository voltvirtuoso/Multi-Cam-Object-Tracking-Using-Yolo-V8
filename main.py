import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import random
import urllib.request
import os
from tqdm import tqdm

# Define paths and URLs
weights_path = "yolov8s.pt"
weights_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/"+weights_path

def download_weights(url, destination):
    """Download weights if they do not exist with a progress bar."""
    if not os.path.exists(destination):
        print(f"Weights not found. Attempting auto download.")
        print(f"Downloading weights from {url}...")
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def report_progress(block_num, block_size, total_size):
                downloaded_size = block_num * block_size
                pbar.update(downloaded_size - pbar.n)
            urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("Download complete.")

# Download YOLOv8 weights if not present
download_weights(weights_url, weights_path)

# Load YOLOv8 model
model = YOLO(weights_path)

# Initialize SORT tracker for each stream
tracker_cam1 = Sort()
tracker_cam2 = Sort()

def detect_objects(frame):
    """Detect objects in the frame using YOLOv8."""
    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append({
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'confidence': score,
                'class_id': int(class_id)
            })
    return detections

def track_objects(frame, detected_objects, tracker):
    """Track objects using SORT tracker."""
    detections = []
    for obj in detected_objects:
        if obj['class_id'] in [0, 2, 5, 7]:  # Filter for relevant classes (e.g., cars, trucks)
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            detections.append([x1, y1, x2, y2, obj['confidence']])
    if detections:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))
    tracked_objects = tracker.update(detections)
    return tracked_objects

def get_unique_color(track_id):
    """Generate a unique color for each track_id."""
    random.seed(track_id)
    return tuple(random.randint(0, 255) for _ in range(3))

def match_objects(objects_cam1, objects_cam2):
    """Match objects between Camera 1 and Camera 2."""
    matched_objects = {}
    for obj1 in objects_cam1:
        x1, y1, x2, y2, id1 = obj1
        for obj2 in objects_cam2:
            x1_, y1_, x2_, y2_, id2 = obj2
            if abs(x1 - x1_) < 50 and abs(y1 - y1_) < 50:
                matched_objects[id1] = id2
    return matched_objects

def draw_tracking(frame, tracked_objects, track_id_map):
    """Draw tracking information on the frame."""
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        color = get_unique_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(track_id_map.get(track_id, track_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Open video streams
cap_cam1 = cv2.VideoCapture("hw_1.mp4")  # Replace with your first camera stream
cap_cam2 = cv2.VideoCapture("hw_2.mp4")  # Replace with your second camera stream

# Output video setup
frame_width = 1280
frame_height = 720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_cam1 = cv2.VideoWriter('output_cam1.avi', fourcc, 30.0, (frame_width, frame_height))
out_cam2 = cv2.VideoWriter('output_cam2.avi', fourcc, 30.0, (frame_width, frame_height))

object_paths = {}  # Track paths across streams
track_id_map = {}  # Map for consistent tracking IDs across cameras

while True:
    ret1, frame1 = cap_cam1.read()
    ret2, frame2 = cap_cam2.read()

    if not ret1 or not ret2:
        break

    # Resize frames to fit screen
    frame1 = cv2.resize(frame1, (frame_width, frame_height))
    frame2 = cv2.resize(frame2, (frame_width, frame_height))

    # Detect objects
    detections_cam1 = detect_objects(frame1)
    detections_cam2 = detect_objects(frame2)

    # Track objects
    tracked_objects_cam1 = track_objects(frame1, detections_cam1, tracker_cam1)
    tracked_objects_cam2 = track_objects(frame2, detections_cam2, tracker_cam2)

    # Match objects between cameras
    matched_objects = match_objects(
        [(int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])) for obj in tracked_objects_cam1],
        [(int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])) for obj in tracked_objects_cam2]
    )

    # Update track_id_map for consistent IDs
    for id1, id2 in matched_objects.items():
        track_id_map[id2] = id1

    # Draw tracking information
    draw_tracking(frame1, tracked_objects_cam1, track_id_map)
    draw_tracking(frame2, tracked_objects_cam2, track_id_map)

    # Write frames to video files
    out_cam1.write(frame1)
    out_cam2.write(frame2)

    # Display frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_cam1.release()
cap_cam2.release()
out_cam1.release()
out_cam2.release()
cv2.destroyAllWindows()
