# packages
from ultralytics import YOLO
import os
import ffmpeg
import cv2
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np

# set working dir
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

# define model
model = YOLO('runs/detect/train_colab6/weights/best.pt') 

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
    # define as args so easier to map pool processes
    video_path, start_frame, end_frame = args

    # video object 
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
            
            # default values in case inference does not work
            midpoint = np.nan, np.nan
            release_cords = np.nan, np.nan

            if class_name.lower() == 'ball' and confidence > 0.4:
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

                    if class_name.lower() == 'ball' and confidence > 0.30: # model has a harder time with the later frame pitches hence the lower confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0]
                        midpoint_x = (x1 + x2) / 2
                        midpoint_y = (y1 + y2) / 2
                        midpoint = (int(midpoint_x), int(midpoint_y))
                    else:
                        continue
                
                # video end and jump
                video_jump = timestamp_ms + 3800
                video_end_ms = timestamp_ms + 4000

                # video info
                video_add = {
                    'release': timestamp_ms / 1000,
                    'video_start': (timestamp_ms - 2000) / 1000,
                    'video_end': video_end_ms,
                    'release_point': release_cords,
                    'tunnel_point': midpoint
                }
                video_intervals.append(video_add)
                
                # jump forward to a new frame
                new_frame_count = int((video_jump / 1000) * frame_rate)
                frame_count = new_frame_count
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
    release_times = df['release'] *1000

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

# find tunneling pairs
def tunneling(release):
    # in case of csv import
    df = release
    
    # extract info from release
    tunnel_points = df['tunnel_point'].astype(str).str.strip('()').str.split(',')
    tunnel_points = [tuple(map(float, map(str.strip, point))) for point in tunnel_points]
    video_numbers = df['video'].tolist()
    release_points = df['release_point'].tolist()

    # eucledian distance
    def get_distance_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    # brute iterate to find closest for each point (does not need to be more efficent)
    def find_closest_for_each(points, video_nums):
        n = len(points)
        results = []
        for i in range(n):
            min_dist_sq = float('inf')
            closest_idx = None

            for j in range(n):
                if i == j:
                    continue
                d_sq = get_distance_sq(points[i], points[j])
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    closest_idx = j
                else:
                    j += 1

            # to account for na's
            if closest_idx is not None:
                results.append({
                    "release_point": release_points[closest_idx],
                    "tunnel_point": points[i],
                    "point_video": video_nums[i],
                    "closest": points[closest_idx],
                    "closest_video": video_nums[closest_idx],
                    "distance_sq": min_dist_sq
                })
            
            else:
                results.append({
                    "release_point": None,
                    "tunnel_point": points[i],
                    "point_video": video_nums[i],
                    "closest": None,
                    "closest_video": None,
                    "distance_sq": None
            })
            
        return results

    closest_results = find_closest_for_each(tunnel_points, video_numbers)
    return closest_results

# splice video at start and end
def video_splice(source_vid, release, root, fps):
    df = release

    # video base
    base = Path(source_vid)
    base = base.stem
    output_dir = os.path.join(root, base)
    os.makedirs(output_dir, exist_ok=True)

    # split videos at specified start and end
    for _, row in df.iterrows():
        # video objec
        cap = cv2.VideoCapture(source_vid)

        # release info
        start = row['video_start']
        end = row['video_end'] / 1000 # end is in ms
        num = row['video']

        # video info 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # save num
        save_num = f"{num}.mp4"

        # create output path
        output_path = os.path.join(output_dir, save_num)
        
        # video writer object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # start and end
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # write to the video file
        current_frame_num = start_frame
        while cap.isOpened() and current_frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            current_frame_num += 1
        cap.release()
        out.release()

    # tunneling pairs
    video_pairs = tunneling(df)
    export = pd.DataFrame(video_pairs)
    export.to_csv(os.path.join(output_dir, "pairs.csv"))

# parallel processing
def process_video_parallel(video_path, fps, video_output):
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
    
    # change to pd 
    release_df = pd.DataFrame(sorted_intervals)
    print(release_df)

    # extract frames
    print(f"Extracting Release Frames from {video_path}")
    extract_release_frames(video_path=video_path, release=release_df)
    
    # find tunneling pairs and create videos
    print(f"Finding Pairs and Creating Videos")
    video_splice(source_vid=video_path, release=release_df, fps = fps, root=video_output)

    return release_df

# output loop
if __name__ == '__main__':
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/test_sets/full_length_test/Tennesse_8_04_f5.mp4"
    output_root = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/release_videos/"
    # release = pd.read_csv("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/release_frames/E_Beneco.csv")
    df = process_video_parallel(video_path=video_dir, fps=30, video_output=output_root)
    csv = pd.DataFrame(df)
    csv.to_csv('./datasets/release_frames/Tennesse_8_04_f5.csv')
    # video_splice(source_vid=video_dir, release=release, root=output_root, fps=30)
