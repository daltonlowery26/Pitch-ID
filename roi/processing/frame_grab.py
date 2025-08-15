import cv2
import pandas as pd
import os

def extract_release_frames(video_path, csv_path):
    output_dir = "release_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        df = pd.read_csv(csv_path)
        release_times = df['release']
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    except KeyError:
        print(f"Error: CSV file must have a column named 'release'.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

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
            print(f"Warning: Could not read frame at {time_ms}ms.")

    cap.release()
    print(f"\nDone. Extracted {saved_count} frames to the '{output_dir}' directory.")

if __name__ == '__main__':
    pitcher1 = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher1.mp4"
    pitcher2 = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/full_length_test/pitcher2.mp4"
    pitcher1csv = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/pitcher1.csv"
    pitcher2csv = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/pitcher2.csv"

    extract_release_frames(pitcher1, pitcher1csv)
    extract_release_frames(pitcher2, pitcher2csv)
