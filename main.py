# main.py
import cv2
from person_detecter import person_detecter
from person_tracker import person_tracker
from pose_classifier import pose_classifier

def main(video_path = 'media1.mp4'):
    cap = cv2.VideoCapture(video_path)
    detector = person_detecter()
    tracker = person_tracker()
    classifier = pose_classifier() 
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.process_frame(frame)

        for detection in detections:
            person_id = detection['id']
            keypoints = detection['keypoints']
            
            pose = classifier.classify_pose(person_id, detection['bbox'])
            
            tracker.update(person_id, pose)

            x1, y1, x2, y2 = detection['bbox']

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            for kp in keypoints:
                cx, cy = int(kp[0]), int(kp[1])
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            
            pose_duration = tracker.get_pose_duration(person_id)

            cv2.putText(frame, f"ID: {person_id} Pose: {pose} ({pose_duration:.1f}s)",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Person Detection & Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
