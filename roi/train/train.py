from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import numpy as np

# weighted dataset because of class imbalances
class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
              weights.append(1)
              continue
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))




if __name__ == '__main__': # so though a conflict isnt caused with pytorch
    
    # specify to use weighted dataset
    build.YOLODataset = YOLOWeightedDataset

    # data
    data_yaml = 'C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/Pitch-ID-ROI-8/data.yaml'

    # model
    model = YOLO("yolov8m.pt")

    # model results
    results = model.train(
    data=data_yaml,
    imgsz=736,
    epochs=500,  
    patience=20,    
    batch = 8,       
    device= 0,
    workers = 2,
    optimizer= 'AdamW'
    )
