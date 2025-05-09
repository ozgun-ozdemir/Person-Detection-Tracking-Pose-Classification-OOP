# pose_classifier.py
import numpy as np

class pose_classifier:
    def __init__(self, movement_threshold = 1, lying_ratio_threshold = 0.7, crouching_ratio_threshold = 1.6):
        self.prev_centers = {}  
        self.movement_threshold = movement_threshold # movement threshold in pixels
        self.lying_ratio_threshold = lying_ratio_threshold # height/width ratio for lying
        self.crouching_ratio_threshold = crouching_ratio_threshold # height/width ratio for crouching

    def classify_pose(self, person_id, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        ratio = height / (width + 1e-6) 

        movement = 0
        if person_id in self.prev_centers:
            prev_cx, prev_cy = self.prev_centers[person_id]
            movement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        self.prev_centers[person_id] = (cx, cy)

        if ratio < self.lying_ratio_threshold:
            pose = "lying"
        elif ratio < self.crouching_ratio_threshold:
            pose = "crouching"
        elif movement > self.movement_threshold:
            pose = "moving"
        else:
            pose = "standing"

        # Debug info
        print(f"ID {person_id} | Ratio: {ratio:.2f} | Movement: {movement:.1f} | Pose: {pose}")

        return pose
    