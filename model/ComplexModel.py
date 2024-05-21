import os
from pathlib import Path

from model.ClassifyModel import DenseModel, InceptionModel, YoloModel
from model.ProcessModel import DirModel, VideoModel
from model.YoloDetect import YoloDetect


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
        self._model_classify = None
        self._color_coef = 1
        self._contrast_coef = 1
        self._bright_coef = 1

    @property
    def color_coef(self):
        return self._color_coef

    @color_coef.setter
    def color_coef(self, new):
        self._color_coef = new

    @property
    def contrast_coef(self):
        return self._contrast_coef

    @contrast_coef.setter
    def contrast_coef(self, new):
        self._contrast_coef = new

    @property
    def bright_coef(self):
        return self._bright_coef

    @bright_coef.setter
    def bright_coef(self, new):
        self._bright_coef = new


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

    @property
    def model_classify(self):
        return self._model_classify
    @model_classify.setter
    def model_classify(self,new):
        self._model_classify = new

    def change_parameters(self, imgsz, path, conf, iou,classify_conf,bright_coef,contrast_coef,color_coef):

        self.path_weights = path if path!='' else r'..\weights\yoloObjDetect\yolo_w.pt'
        self.imgsz = imgsz if imgsz != -1 else 640
        self.conf = conf if conf != -1 else 0.25
        self.iou = iou if iou !=-1 else 0.7
        self.classify_conf = classify_conf if classify_conf!=-1 else 0.6
        self.bright_coef = bright_coef if bright_coef != -1 else 1
        self.contrast_coef = contrast_coef if contrast_coef !=-1 else 1
        self.color_coef = color_coef if color_coef !=-1 else 1

        print(self.path_weights, self.imgsz, self.conf, self.iou,self.classify_conf,self.bright_coef,self.contrast_coef,self.color_coef)

    def create_model(self,classify_model_choose="inception"):

        if classify_model_choose=="dense":
            self.model_classify = DenseModel(os.path.join(os.path.dirname(__file__), r'..\weights\denseClassify\dense_w.pt'),self._classify_conf)
        elif classify_model_choose=="inception":
            self.model_classify = InceptionModel(
                os.path.join(os.path.dirname(__file__), r'..\weights\inceptionClassify\inception_w.pt'), self._classify_conf)
        elif classify_model_choose=="yolo":
            self.model_classify = YoloModel(
                os.path.join(os.path.dirname(__file__), r'..\weights\yoloClassify\yolo_classify_w.pt'), self._classify_conf)
        self.model_classify.createModel()
        self.model = YoloDetect()

    def createSaveDir(self):
        saved_runs = os.listdir(os.path.join(os.path.dirname(__file__), r'..\runs'))
        if (not len(saved_runs)):
            save_path = os.path.join(os.path.dirname(__file__), r'..\runs\trolley1')
        else:
            max_number =1
            for saved_run in saved_runs:
                if int(saved_run.split("trolley")[-1])>max_number:
                    max_number = int(saved_run.split("trolley")[-1])
            max_number+=1
            save_path = os.path.join(os.path.dirname(__file__), r'..\runs\trolley' + str(max_number))
        os.mkdir(save_path)
        os.mkdir(save_path + '\\detect')
        os.mkdir(save_path + '\\classify')
        os.mkdir(save_path + '\\result')
        return save_path

    def passParameters(self):
        self.model_classify.classify_conf = self.classify_conf
        self.model_classify.color_coef = self.color_coef
        self.model_classify.contrast_coef = self.contrast_coef
        self.model_classify.bright_coef = self.bright_coef
        self.model.path_weights = self.path_weights
        self.model.imgsz = self.imgsz
        self.model.conf = self.conf
        self.model.iou = self.iou

    def process(self, path,showRegime):
        save_path = self.createSaveDir()
        self.passParameters()
        if showRegime==1:
            main_model = DirModel(self.model,self.model_classify,path,save_path)
        else:
            main_model = VideoModel(self.model,self.model_classify,path,save_path)
        main_model.process()



