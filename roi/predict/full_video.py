# packages
from ultralytics import YOLO
import os
import cv2
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np

# set working dir
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

# process the video
def process_segment(args):
    video_path, start_frame, end_frame = args
    
    model = YOLO('runs/detect/train_colab4/weights/best.pt')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video in process for frames {start_frame}-{end_frame}.")
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
            midpoint = np.nan, np.nan
            release_cords = np.nan, np.nan

            if class_name.lower() == 'ball' and confidence > 0.45:
                # time of release
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"Ball recognized at: {timestamp_ms / 1000:.2f} seconds (from seg starting at frame {start_frame})")
                
                # release cords
                x1, y1, x2, y2 = box.xyxy[0]
                release_x = (x1 + x2) / 2
                release_y = (y1 + y2) / 2
                release_cords = (int(release_x), int(release_y))
                
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

                    if class_name.lower() == 'ball' and confidence > 0.25: # model has a harder time with the later frame pitches hence the lower confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0]
                        midpoint_x = (x1 + x2) / 2
                        midpoint_y = (y1 + y2) / 2
                        midpoint = (int(midpoint_x), int(midpoint_y))
                    else:
                        continue
                
                # video end and jump
                video_end_ms = timestamp_ms + 3500
                # video info
                video_add = {
                    'release': timestamp_ms / 1000,
                    'video_start': (timestamp_ms - 1000) / 1000,
                    'video_end': video_end_ms,
                    'release_point': release_cords,
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

# generate release frames
def extract_release_frames(video_path, release):

    # make path
    root_dir = "./datasets/release_frames/"
    video_name = Path(video_path)
    video_name = video_name.stem
    output_dir = os.path.join(root_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # extract the release times
    df = release
    release_times = df['release']

    # video object
    cap = cv2.VideoCapture(video_path)


    saved_count = 0
    for i, time_ms in enumerate(release_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        success, frame = cap.read()

        if success:
            frame_filename = f"frame_at_{int(time_ms)}ms.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1
        else:
            print(f"Could not read frame at {time_ms}ms.")

    cap.release()

def process_video_parallel(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return []

    # amount of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_third = total_frames // 3
    second_third = 2 * total_frames // 3
    cap.release()

    # split video into three parts
    tasks = [
        (video_path, 0, first_third),
        (video_path, first_third, second_third),
        (video_path, second_third, total_frames)
    ]

    # para processing
    with Pool(processes=3) as pool:
        results_from_processes = pool.map(process_segment, tasks)

    # reorder videos based on time of release
    all_video_intervals = []
    for interval_list in results_from_processes:
        all_video_intervals.extend(interval_list)
    sorted_intervals = sorted(all_video_intervals, key=lambda x: x['release'])
    for i, interval in enumerate(sorted_intervals):
        interval['video'] = i + 1
    
    # change to pd for 
    release_df = pd.DataFrame(sorted_intervals)

    # extract frames
    print("extracting release frames")
    extract_release_frames(video_path=video_path, release=release_df)
    
    return release_df

# main
if __name__ == '__main__':
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher2_splice.mp4"
    df = process_video_parallel(video_dir)
    csv = pd.DataFrame(df)
    csv.to_csv("split.csv")
