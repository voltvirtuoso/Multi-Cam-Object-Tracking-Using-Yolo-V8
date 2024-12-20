# Multi-Cam Object Tracking Using YOLOv8

This repository contains a multi-camera object detection and tracking system implemented using the YOLOv8 model for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking. The project processes video streams to detect and track objects across multiple cameras, maintaining consistent object IDs.

---

## Features
- **YOLOv8 Object Detection**: Accurately detects objects like cars and trucks in video streams.
- **SORT Tracking**: Assigns unique IDs for object tracking across frames.
- **Multi-Camera Matching**: Matches and maintains object IDs across multiple cameras.
- **Customizable Object Classes**: Supports tracking specific classes of objects (e.g., cars, buses, trucks).
- **Video Output**: Generates annotated output videos with bounding boxes and track IDs.
- **Auto Weight Download**: Automatically downloads YOLOv8 weights if missing.

---

## Table of Contents
1. [Installation](#installation)
2. [Requirements](#requirements)
3. [How It Works](#how-it-works)
4. [Usage](#usage)
5. [Customization](#customization)
6. [Output](#output)
7. [Acknowledgements](#acknowledgements)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/voltvirtuoso/Multi-Cam-Object-Tracking-Using-Yolo-V8.git
   cd Multi-Cam-Object-Tracking-Using-Yolo-V8
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLO
- TQDM (for progress bars)
- SORT Tracker

Ensure the video files (`hw_1.mp4` and `hw_2.mp4`) are present in the project directory. Replace these with your own video files or camera streams as needed.

---

## How It Works

1. **Weight Downloading**: Automatically downloads YOLO weights if not found and source link is provided.
2. **Object Detection**: YOLOv8 detects objects in each frame of the video.
3. **Tracking**: SORT assigns unique IDs for consistent object tracking.
4. **Multi-Camera Matching**: Matches objects detected in Camera 1 and Camera 2 based on spatial proximity.
5. **Visualization**: Draws bounding boxes, class labels, and track IDs on video frames.
6. **Output**: Saves annotated videos to disk and displays real-time tracking.

---

## Usage

1. **Run the Main Script**:
   ```bash
   python main.py
   ```

2. **Input Video Streams**:
   - The script processes two input streams: `hw_1.mp4` and `hw_2.mp4`.
   - Replace these with your video file paths or live camera streams.

3. **View Outputs**:
   - Annotated videos are saved as `output_cam1.avi` and `output_cam2.avi`.
   - Real-time visualizations are displayed in OpenCV windows.

---

## Customization

### Classes to Detect
Update the `track_objects()` function in `main.py` to track other classes. By default, the script tracks:
- **Class 0**: Person
- **Class 2**: Car
- **Class 5**: Bus
- **Class 7**: Truck

For additional classes, refer to the [COCO Dataset Class List](https://github.com/ultralytics/yolov5/blob/master/data/coco.names).

### Video Resolution
Modify `frame_width` and `frame_height` variables in `main.py` to change the resolution of processed videos.

### Camera Streams
Replace the default video streams (`hw_1.mp4` and `hw_2.mp4`) with your own file paths or camera URLs.

---

## Output

1. **Video Files**:
   - `output_cam1.avi`: Annotated output for Camera 1.
   - `output_cam2.avi`: Annotated output for Camera 2.

2. **Annotations**:
   - Bounding boxes for detected objects.
   - Class labels and confidence scores.
   - Consistent track IDs across frames and cameras.

---

## Acknowledgements

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: For the state-of-the-art YOLOv8 model.
- **SORT Tracker**: For efficient object tracking.
- **OpenCV**: For video processing and visualization.

---

Feel free to fork and contribute to this project. If you have any questions or suggestions, please open an issue or submit a pull request. 🚀
