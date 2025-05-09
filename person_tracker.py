# person_tracker.py
import time

class person_tracker:
    def __init__(self):
        self.persons = {} 

    def update(self, person_id, pose):
        now = time.time()
        
        if person_id not in self.persons:
            self.persons[person_id] = {
                'pose': pose,
                'last_update': now,
                'pose_durations': {'lying': 0, 'crouching':0, 'moving': 0, 'standing': 0}  
            }

        person = self.persons[person_id]
        elapsed = now - person['last_update']
        
        person['pose_durations'][pose] += elapsed
        person['pose'] = pose
        person['last_update'] = now

    def get_pose_duration(self, person_id):
        if person_id in self.persons:
            person = self.persons[person_id]
            return person['pose_durations'].get(person['pose'], 0) 
        return 0
