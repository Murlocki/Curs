from ultralytics import YOLO
import os
from pathlib import Path

import cv2

class Model:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def process(self, path, imgsz, conf, iou):
        pass

    def write_log(self):
        pass


class ModelOnePng(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)

    def process(self, path, imgsz, conf, iou):
        result = self.model.predict(source=path, imgsz=imgsz, conf=conf, iou=iou,save=True,save_txt=True, project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect')


    def write_log(self):
        pass


class ModelMultiple(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)


class ModelOneVideo(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)


class YoloModel:
    def __init__(self, imgsz=640, path=os.path.join(os.path.dirname(__file__), r'..\weights\yolo_w.pt'), conf=0.25, iou = 0.7):
        self.path_weights = path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model = None
    def change_parameters(self, imgsz, path, conf, iou):
        if path != '':
            self.path_weights = path
        else:
            self.path_weights = r'..\weights\yolo_w.pt'

        if imgsz != -1:
            self.imgsz = imgsz
        else:
            self.imgsz = 640

        if conf != -1:
            self.conf = conf
        else:
            self.conf = 0.25

        if iou != -1:
            self.iou = iou
        else:
            self.iou = 0.7
        print(self.path_weights, self.imgsz, self.conf, self.iou)

    def create_model(self, path, number):
        if self.model is not None:
            del self.model

        file = Path(path)
        if file.is_file():
            if number == 1:
                self.model = ModelOnePng(self.path_weights)
            elif number == 2:
                self.model = ModelOneVideo(self.path_weights)

        return self
    def process(self,path):
        self.model.process(path, self.imgsz, self.conf, self.iou)
        return self
