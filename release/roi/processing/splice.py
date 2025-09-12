import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            frame_filename = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1


    cap.release()
    print(f"Extracted {saved_count} frames from {os.path.basename(video_path)}.")


video_root = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/labeling sets/video/LHH_9_2/"
photo_root = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/labeling sets/LHH_9_2_frames"
video_extensions = ('.mp4')


for filename in os.listdir(video_root):
    if filename.lower().endswith(video_extensions):
        video_path = os.path.join(video_root, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = photo_root

        print(f"Processing '{filename}'...")
        extract_frames(video_path=video_path, output_dir=output_dir)

print("All videos processed.")