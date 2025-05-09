# Person Detection, Tracking & Pose Classification (OOP)

This is a real-time computer vision application that detects people in a video, tracks them across frames, classifies their body pose, and keeps track of how long each person remains in a given pose.

## Features

- Person detection using YOLOv11n Pose model
- Unique person tracking across frames
- Pose classification based on bounding box aspect ratio and motion
- Real-time pose duration tracking
- Modular structure for easy extension

## Modules

- `main.py`            | Runs the pipeline: detection - tracking - classification - visualization |
- `person_detecter.py` | Detects people and keypoints using YOLOv11n pose model |
- `person_tracker.py`  | Tracks individual person states and pose durations |
- `pose_classifier.py` | Classifies pose based on bounding box ratio and center movement |

## Requirements

- `Python 3.x`
- `OpenCV`
- `YOLOv11`
- `numpy`
