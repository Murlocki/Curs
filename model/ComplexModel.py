import os
from pathlib import Path


#Класс составной модели
class ComplexModel:
    def __init__(self, imgsz=640, path=os.path.join(os.path.dirname(__file__), r'..\weights\yoloObjDetect\yolo_w.pt'), conf=0.25,
                 iou=0.7,classify_conf=0.6):
        self._path_weights = path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._classify_conf = classify_conf
        self._model = None

    @property
    def classify_conf(self):
        return self._classify_conf
    @classify_conf.setter
    def classify_conf(self,new):
        self._classify_conf=new

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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new):
        self._model = new

    def change_parameters(self, imgsz, path, conf, iou,classify_conf):
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

        if classify_conf!=-1:
            self.classify_conf = classify_conf
        else:
            self.classify_conf=0.6

        print(self.path_weights, self.imgsz, self.conf, self.iou,self.classify_conf)
        return self

    def create_model(self, path, number):
        file = Path(path)
        # if file.is_file():
        #     if number == 1:
        #         self.model = ModelOnePng(self.path_weights)
        #     elif number == 2:
        #         self.model = ModelShowVideo(self.path_weights)
        # elif file.is_dir():
        #     self.model = ModelMultiple(self.path_weights)
        # return self

    def process(self, path):
        pass
        # self.model = self.model.process(path, self.imgsz, self.conf, self.iou)
        # return self