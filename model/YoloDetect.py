from abc import ABC, abstractmethod
from ultralytics import YOLO
import os
from pathlib import Path
import sys
import cv2


class YoloDetect():
    def __init__(self,imgsz=640, path=os.path.join(os.path.dirname(__file__), r'..\weights\yoloObjDetect\yolo_w.pt'), conf=0.25,
                 iou=0.7):
        self._path_weights = path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._model = None

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,new):
        self._model = new

    @property
    def path_weights(self):
        return self._path_weights

    @path_weights.setter
    def path_weights(self, new):
        self._path_weights = new

    @property
    def imgsz(self):
        return self._imgsz

    @imgsz.setter
    def imgsz(self, new):
        self._imgsz = new

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, new):
        self._conf = new

    @property
    def iou(self):
        return self._iou

    @iou.setter
    def iou(self, new):
        self._iou = new
    def process(self,process_path,save_path=""):
        self._model = YOLO(self.path_weights)
        results = self.model(source=process_path,
                            imgsz=self.imgsz,
                            conf=self.conf,
                            iou=self.iou,
                            save=True if save_path!="" else False,
                            save_txt=True,
                            project=save_path+"\\detect",
                            name='detect',
                            vid_stride=5)
        return results



