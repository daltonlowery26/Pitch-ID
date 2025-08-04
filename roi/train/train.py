from ultralytics import YOLO

if __name__ == '__main__': # so though a conflict isnt caused with pytorch

    # data
    data_yaml = 'C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/Pitch-ID-ROI-5/data.yaml'

    # model
    model = YOLO("yolov8m.pt")

    # model results
    results = model.train(
    data=data_yaml,
    imgsz=640,
    epochs=500,  
    patience=20,    
    batch = 12,       
    device= 0,
    workers = 2,
    optimizer= 'AdamW'
    )
