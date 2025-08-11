from ultralytics import YOLO
import os
import cv2
import pandas as pd

os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

def pred_loop(video_path):
    model = YOLO('runs/detect/train_colab/weights/best.pt')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    ball_first_appearance = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if success:
            target_size = (1024, 1024)
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frame = resized_frame

        results = model(frame, device = 0)
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() == 'ball':
                if not ball_first_appearance and confidence > 0.5:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp_s = timestamp_ms / 1000
                    print(f"Ball first recognized at: {timestamp_s:.2f} seconds")
                    
                    cap.release()
                    return timestamp_s, confidence
                

if __name__ == '__main__':
    elite_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/cut_test/"
    release = []
    for path in os.listdir(elite_dir):
        search = os.path.join(elite_dir, path)
        time = pred_loop(search)
        data = {'file': path, 'time': time}
        release.append(data)
    df = pd.DataFrame(release)
    df.to_csv('testing2.csv')