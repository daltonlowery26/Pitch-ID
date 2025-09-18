import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# set working dir
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/pitch_type')
# define model
model = YOLO('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/runs/detect/train_colab6/weights/best.pt') 

# extract cord path
def process_segment(args):
    # frame resize
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
    
    # define as args so easier to map pool processes
    dir_path = args

    export = []

    for maindir, dirnames, filenames in os.walk(dir_path):
        
        for file in filenames:
            print(file)
            # get video
            video = os.path.join(maindir, file)

            # video object 
            cap = cv2.VideoCapture(video)

            if not cap.isOpened():
                print(f"Could not open video")
                return []

            frame_count = 0

            while frame_count < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
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
                        frame_count_release = 1
                        
                        frame_cords = []
                        while frame_count_release <= 8:
                            # find location of ball at decison point
                            tunneling_frame = frame_count + frame_count_release
                            cap.set(cv2.CAP_PROP_FRAME_COUNT, tunneling_frame)
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
                                    tunnel_info = {
                                        f'midpoint{frame_count_release}':midpoint
                                    }
                                    frame_cords.append(tunnel_info)
                                else:
                                    continue
                            frame_count_release += 1

                        data = {
                            'file':maindir,
                            'ball cords': frame_cords
                            }
                        
                        export.append(data)
                        frame_count = 1000000 # break main while loop because using spliced video
                        break
    # export pd
    ex_pd = pd.DataFrame(export)
    ex_pd.to_csv('locations.csv')

if __name__ == "__main__":
    dir = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/pitch_type/data_sets/"
    process_segment(dir)