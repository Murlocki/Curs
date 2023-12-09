from abc import ABC, abstractmethod
from ultralytics import YOLO
import os
from pathlib import Path
import sys
import cv2

class Model(ABC,object):
    def __init__(self, weights_path):
        self._model = YOLO(weights_path)

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,new):
        self._model = new


    @abstractmethod
    def process(self, path, imgsz, conf, iou):
        pass

    @abstractmethod
    def write_log(self,results):
        pass


class ModelOnePng(Model):


    def process(self, path, imgsz, conf, iou):
        result = self.model(source=path, imgsz=imgsz, conf=conf, iou=iou,save=True,save_txt=True, project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect',vid_stride=5)
        self.write_log(results=result)
        self.show_imgs(result)
        return self
    def write_log(self,results):
        with open(os.path.join(os.path.dirname(__file__), '..\logs\logs.txt'), 'a') as file:
            for res in results[0]:
                boxes = res.boxes.cpu().numpy()
                file.write(f'{boxes.cls[0]} {boxes.xywhn[0][0]} {boxes.xywhn[0][1]} {boxes.xywhn[0][2]} {boxes.xywhn[0][3]}\n')
                print(boxes.cls, boxes.xywhn)
        return self
    def show_imgs(self,results):
        i = Path(results[0].path).name
        i=i.replace('.mp4','.avi')
        print(i)
        cap = cv2.VideoCapture(results[0].save_dir+r'\\'+i)
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        return self

class ModelMultiple(Model):


    def process(self, path, imgsz, conf, iou):
        result = self.model(source=path, imgsz=imgsz, conf=conf, iou=iou,save=True,save_txt=True, project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect')
        self.write_log(results=result)
        self.show_imgs(result)
        return self
    def write_log(self,results):
        with open(os.path.join(os.path.dirname(__file__), '..\logs\logs.txt'), 'a') as file:
            for res in results:
                print(res)
                for r in res:
                    boxes = r.boxes.cpu().numpy()
                    print(f'{boxes.cls[0]} {boxes.xywhn[0][0]} {boxes.xywhn[0][1]} {boxes.xywhn[0][2]} {boxes.xywhn[0][3]}\n')
                    file.write(f'{boxes.cls[0]} {boxes.xywhn[0][0]} {boxes.xywhn[0][1]} {boxes.xywhn[0][2]} {boxes.xywhn[0][3]}\n')
                    print(boxes.cls, boxes.xywhn)
        return self
    def show_imgs(self,results):
        import webbrowser
        path = results[0].save_dir+r'\\'
        webbrowser.open(path)
        return self
class ModelShowVideo(Model):


    def process(self, path, imgsz, conf, iou):
        cap = cv2.VideoCapture(path)
        ret = True
        time_skips = float(150)
        count = 0
        while ret:
            ret,frame = cap.read()
            if ret:
                result = self.model.predict(frame, imgsz=imgsz, conf=conf, iou=iou,
                                  project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect')
                cap.set(cv2.CAP_PROP_POS_MSEC,(count * time_skips))
                count+=1
                frame_ = result[0].plot()
                cv2.imshow('frame',frame_)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
        return self

    def write_log(self,results):
        pass
class YoloModel:
    def __init__(self, imgsz=640, path=os.path.join(os.path.dirname(__file__), r'..\weights\yolo_w.pt'), conf=0.25, iou = 0.7):
        self._path_weights = path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._model = None

    @property
    def path_weights(self):
        return self._path_weights
    @path_weights.setter
    def path_weights(self,new):
        self._path_weights = new

    @property
    def imgsz(self):
        return self._imgsz
    @imgsz.setter
    def imgsz(self,new):
        self._imgsz = new

    @property
    def conf(self):
        return self._conf
    @conf.setter
    def conf(self,new):
        self._conf = new

    @property
    def iou(self):
        return self._iou
    @iou.setter
    def iou(self,new):
        self._iou = new

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,new):
        self._model = new

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
        return self
    def create_model(self, path, number):
        file = Path(path)
        if file.is_file():
            if number == 1:
                self.model = ModelOnePng(self.path_weights)
            elif number == 2:
                self.model = ModelShowVideo(self.path_weights)
        elif file.is_dir():
            self.model = ModelMultiple(self.path_weights)
        return self
    def process(self,path):
        self.model=self.model.process(path, self.imgsz, self.conf, self.iou)
        return self
