from ultralytics import YOLO
import os
import cv2
from multiprocessing import Pool
import pandas as pd
import time

os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

# resize the frame using padding
def frame_resize(frame, target_size):
    if frame is not None and len(frame.shape) == 3:
        h, w, _ = frame.shape
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_ar = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top_pad = (target_h - new_h) // 2
        bottom_pad = target_h - new_h - top_pad
        left_pad = (target_w - new_w) // 2
        right_pad = target_w - new_w - left_pad

        resized_frame = cv2.copyMakeBorder(resized_ar, top_pad, bottom_pad, left_pad, right_pad, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return resized_frame

# process the video, in function so para processing can work    
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
        
        resized_frame = frame_resize(frame, (512,512))
        results = model(resized_frame, device=0, verbose=False)
        
        results = results[0]
        ball_detected = False

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            midpoint = 0,0

            if class_name.lower() == 'ball' and confidence > 0.45:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"Ball recognized at: {timestamp_ms / 1000:.2f} seconds (from segment starting at frame {start_frame})")

                # find location of ball at decison point
                tunneling_time = timestamp_ms + 167
                cap.set(cv2.CAP_PROP_POS_MSEC, tunneling_time)
                success, frame = cap.read()
                resized_frame = frame_resize(frame, (512,512))
                results = model(resized_frame, device=0, verbose=False)
                results = results[0]
                
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if class_name.lower() == 'ball' and confidence > 0.25:
                        x1, y1, x2, y2 = box.xyxy[0]
                        midpoint_x = (x1 + x2) / 2
                        midpoint_y = (y1 + y2) / 2
                        midpoint = (int(midpoint_x), int(midpoint_y))
                    else:
                        continue

                video_end_ms = timestamp_ms + 3500
                video_add = {
                    'release': timestamp_ms / 1000,
                    'video_start': (timestamp_ms - 1000) / 1000,
                    'video_end': video_end_ms,
                    'tunnel_point': midpoint
                }
                video_intervals.append(video_add)
                
                # jump forward to a new frame
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
    first_third = total_frames // 3
    second_third = 2 * total_frames // 3
    cap.release()

    tasks = [
        (video_path, 0, first_third),
        (video_path, first_third, second_third),
        (video_path, second_third, total_frames)
    ]
    # split for para processing
    with Pool(processes=3) as pool:
        results_from_processes = pool.map(process_segment, tasks)

    all_video_intervals = []
    for interval_list in results_from_processes:
        all_video_intervals.extend(interval_list)

    sorted_intervals = sorted(all_video_intervals, key=lambda x: x['release'])
    
    for i, interval in enumerate(sorted_intervals):
        interval['video'] = i + 1

    return sorted_intervals

  
if __name__ == '__main__':
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher2_splice.mp4"
    df = process_video_parallel(video_dir)
    csv = pd.DataFrame(df)
    csv.to_csv("split.csv")
