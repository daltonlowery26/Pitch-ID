from ultralytics import YOLO
import os
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/')

model = YOLO('models/train/weights/best.pt')

video_path = "datasets/roi_video/CB_Ball_77.5.mp4"
cap = cv2.VideoCapture(video_path)
ball_first_appearance = False


while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
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
                ball_first_appearance = True
        else:
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

cap.release()
cv2.destroyAllWindows()

print("Video processing finished.")