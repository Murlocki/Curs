from ultralytics import YOLO
import os
class YoloModel():
    def __init__(self,imgsz=640,path=os.path.join(os.path.dirname(__file__), '..\weights\yolo_w.pt'),conf=0.25,uoi=0.7):
        self.path_weights = path
        self.imgsz = imgsz
        self.conf = conf
        self.uoi = uoi

    def change_parameters(self,imgsz,path,conf,uoi):

        if path!='':
            self.path_weights = path
        else:
            self.path_weights = '..\weights\yolo_w.pt'

        if imgsz!=-1:
            self.imgsz = imgsz
        else:
            self.imgsz = 640

        if conf!=-1:
            self.conf = conf
        else:
            self.conf = 0.25

        if uoi!=-1:
            self.uoi = uoi
        else:
            self.uoi = 0.7
        print(self.path_weights,self.imgsz,self.conf,self.uoi)