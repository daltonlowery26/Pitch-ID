from ultralytics import YOLO
import os
import cv2
from multiprocessing import Pool
import pandas as pd
import time

os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

def process_segment(args):
    video_path, start_frame, end_frame = args
    
    model = YOLO('runs/detect/train_colab4/weights/best.pt')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video in process for frames {start_frame}-{end_frame}.")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    video_intervals = []
    frame_count = start_frame

    while frame_count < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()

        if not success:
            break
        
        target_size = (512, 512)
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        results = model(resized_frame, device=0, verbose=False)
        
        results = results[0]
        ball_detected = False

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() == 'ball' and confidence > 0.5:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"Ball recognized at: {timestamp_ms / 1000:.2f} seconds (from segment starting at frame {start_frame})")

                video_end_ms = timestamp_ms + 3500
                video_add = {
                    'release': timestamp_ms / 1000,
                    'video_start': (timestamp_ms - 1000) / 1000,
                    'video_end': video_end_ms 
                }
                video_intervals.append(video_add)


                new_frame_count = int((video_end_ms / 1000) * frame_rate)
                frame_count = min(new_frame_count, end_frame)
                ball_detected = True
                break
        
        if not ball_detected:
            frame_count += 1
            
    cap.release()
    return video_intervals

def process_video_parallel(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    midpoint_frame = total_frames // 2
    cap.release()

    print(f"Total frames: {total_frames}, processing in two halves: 0-{midpoint_frame} and {midpoint_frame}-{total_frames}")

    tasks = [
        (video_path, 0, midpoint_frame),
        (video_path, midpoint_frame, total_frames)
    ]
    with Pool(processes=2) as pool:
        results_from_processes = pool.map(process_segment, tasks)

    all_video_intervals = []
    for interval_list in results_from_processes:
        all_video_intervals.extend(interval_list)

    sorted_intervals = sorted(all_video_intervals, key=lambda x: x['release'])
    
    for i, interval in enumerate(sorted_intervals):
        interval['video'] = i + 1

    return sorted_intervals

def pred_loop(video_path):
    model = YOLO('runs/detect/train_colab4/weights/best.pt')
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
            target_size = (512, 512)
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
    return video_intervals
  
if __name__ == '__main__':
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher2_splice.mp4"
    df = process_video_parallel(video_dir)
    csv = pd.DataFrame(df)
    csv.to_csv("split")
