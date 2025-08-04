from ultralytics import YOLO
import os
import cv2


os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

def pred_loop():

    model = YOLO('runs/detect/train8/weights/best.pt')
    video_path = "datasets/roi_video/Ridings-SM18 (2019)-RHP-LHB-00.04.51.469-00.04.53.815-seg28.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    ball_first_appearance = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

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
                    return
    print("done")

if __name__ == '__main__':
    pred_loop()