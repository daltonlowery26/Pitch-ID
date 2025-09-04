import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import os

os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

def tunneling(release):
    # in case of csv import
    df = release
    
    # extract info from release
    tunnel_points = df['tunnel_point'].str.strip('()').str.split(',')
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


def video_splice(source_vid, release, root):
    df = pd.read_csv(release)

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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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



if __name__ == '__main__':
    # file paths
    video_dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/test_sets/full_length_test/E_Beneco.mp4"
    root = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/release_videos/"
    release = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/release_frames/E_Beneco.csv"

    # splice video
    video_splice(source_vid=video_dir, release=release, root=root)




