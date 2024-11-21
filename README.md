# Person Detection and Tracking with YOLO and ByteTrack

This project is designed for real-time person detection and tracking in video streams using the YOLO model for object detection and ByteTrack for object tracking. The system can count the number of people in a specified zone, and issue a warning if the count exceeds a predefined threshold.

---

## Features

- **Real-time person detection**: Uses the YOLOv8 model for accurate person detection.
- **Object tracking**: Incorporates ByteTrack to track detected persons across frames.
- **Custom zones**: Detects and tracks people within specified zones.
- **Warnings**: Issues a visible warning when the number of people exceeds a certain threshold.

---

## Prerequisites

Make sure you have Python installed (preferably 3.10 or newer). It is recommended to use Conda or a virtual environment for a clean setup.

### Install Required Dependencies

The required Python packages are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt

Required Python Packages

The main dependencies for this project are:
	•	opencv-python
	•	ultralytics
	•	supervision
	•	numpy

Getting Started

1. Clone the Repository

Clone this repository to your local machine:
git clone aayushpe/Person-Detection
cd YOLO-Detection

Prepare the Model

Ensure you have the YOLO model file (e.g., person_ncnn_model) in the project directory. If you do not have it, you can train a custom YOLOv8 model or use a pre-trained one.

3. Run the Project

To start the detection and tracking system, simply run:
```

python person.py
Replace people.mp4 with the path to your video file if using a different input.

How It Works

    1.	Detection: The YOLOv8 model detects persons in each frame of the video.
    2.	Tracking: ByteTrack assigns IDs to detected persons and tracks them across frames.
    3.	Zone Specification: A custom polygon zone is defined to filter the detections within a specific area.
    4.	Warnings: If the total number of people exceeds a threshold (default: 8), a red warning appears at the center of the frame.

Customization

    •	Threshold for Warnings: Modify the threshold for warnings in the display_counts_and_warnings function in person.py.
    •	Bounding Zones: Customize the bounding zone in the bounding_zone variable in person.py.

Acknowledgments

    •	YOLOv8: For the object detection framework.
    •	Supervision: For object tracking and annotation utilities.
    •	OpenCV: For image and video frame processing.
