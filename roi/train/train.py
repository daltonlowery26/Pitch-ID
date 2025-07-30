from ultralytics import YOLO
import os


if __name__ == '__main__': # so though a conflict isnt caused with pytorch

    # data
    data_yaml = 'C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/Pitch-ID-ROI-3/data.yaml'

    # model
    model = YOLO("yolov8m.pt")

    # model results
    results = model.train(
    data=data_yaml,
    imgsz=640,
    epochs=100,  
    patience=10,    
    batch = 8,       
    device= 0,
    workers = 2         
    )
