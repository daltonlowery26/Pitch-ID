from ultralytics import YOLO
import os
import cv2
import pandas as pd
import time
import numpy as np

os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

def pred_loop(video_path):
    model = YOLO('runs/detect/train_colab2/weights/best.pt')
    cap = cv2.VideoCapture(video_path)
    video_intervals = []
    times = []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return video_intervals

    if cap.isOpened():
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    video_count = 0


    while frame_count < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()

        if not success:
            break

        if success:
            target_size = (1024, 1024)
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frame = resized_frame

        timestart = time.time()
        results = model(frame, device = 0, verbose = False)
        timeend = time.time()
        diff = timeend - timestart
        times.append(diff)

        results = results[0]
        ball_detected = False

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() == 'ball' and confidence > 0.5:
                    video_count += 1
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    print(f"Ball recognized at: {timestamp_ms:.2f} seconds")

                    video_end = timestamp_ms + 3500
                    video_add = {'video':video_count, 'release': timestamp_ms / 1000, 'video_start': timestamp_ms - 1000, 'video_end': video_end}
                    video_intervals.append(video_add)

                    frame_count = int((video_end / 1000) * frame_rate)
                    ball_detected = True
                    break
                    
        if not ball_detected:
            frame_count += 1

    cap.release()
    print(np.mean(times))
    return video_intervals
  

if __name__ == '__main__':
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher2_splice.mp4"
    df = pred_loop(video_dir)
    results = pd.DataFrame(df)
    results.to_csv('jose.csv')
    