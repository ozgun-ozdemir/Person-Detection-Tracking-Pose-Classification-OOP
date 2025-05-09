# person_detecter.py
from ultralytics import YOLO

class person_detecter:
    def __init__(self, model_path = "yolo11n-pose.pt", device = "mps"): # MPS (Apple Silicon) | cuda (NVIDIA GPU)
        self.model = YOLO(model_path)
        self.model.to(device)

    def process_frame(self, frame):
        results = self.model.track(
            frame,
            persist = True,
            conf = 0.4, # confidence threshold
            iou = 0.3, # IoU threshold
            classes = [0] # only person class
        )
        
        detections = []
        for r in results:
            if r.boxes.id is not None and r.keypoints is not None:
                ids = r.boxes.id.cpu().numpy().astype(int) 
                keypoints = r.keypoints.xy.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()

                for person_id, kp, box in zip(ids, keypoints, boxes):
                    detections.append({'id': person_id, 'keypoints': kp, 'bbox': box})

        return detections