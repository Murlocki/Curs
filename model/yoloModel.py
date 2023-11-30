from ultralytics import YOLO
import os
from pathlib import Path
import sys
import cv2

class Model:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def process(self, path, imgsz, conf, iou):
        pass

    def write_log(self,results):
        pass


class ModelOnePng(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)

    def process(self, path, imgsz, conf, iou):
        result = self.model(source=path, imgsz=imgsz, conf=conf, iou=iou,save=True,save_txt=True, project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect')
        self.write_log(results=result)
        self.show_imgs(result)
    def write_log(self,results):
        with open(os.path.join(os.path.dirname(__file__), '..\logs\logs.txt'), 'a') as file:
            for res in results[0]:
                boxes = res.boxes.cpu().numpy()
                file.write(f'{boxes.cls[0]} {boxes.xywhn[0][0]} {boxes.xywhn[0][1]} {boxes.xywhn[0][2]} {boxes.xywhn[0][3]}\n')
                print(boxes.cls, boxes.xywhn)
    def show_imgs(self,results):
        i = Path(results[0].path).name
        img = cv2.imread(results[0].save_dir+r'\\'+i, cv2.IMREAD_ANYCOLOR)

        while True:
            cv2.imshow("result", img)
            cv2.waitKey(0)
            sys.exit()  # to exit from all the processes


class ModelMultiple(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)


class ModelOneVideo(Model):
    def __init__(self, weights_path):
        super().__init__(weights_path)

    def process(self, path, imgsz, conf, iou):
        results = self.model.track(path, imgsz=imgsz, conf=conf, iou=iou, save=True, save_txt=True,
                                  project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect',
                                  stream=True)
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
            print(boxes,masks,probs)
        #cap = cv2.VideoCapture(path)
        #ret = True
        #while ret:
            #ret,frame = cap.read()
            #if ret:
                #result = self.model.track(frame, imgsz=imgsz, conf=conf, iou=iou,
                                  #project=os.path.join(os.path.dirname(__file__), r'..\runs'), name='detect',persist=True)
                #frame_=result[0].plot()
                #cv2.imshow('frame',frame_)
                #if cv2.waitKey(25) & 0xFF==ord('q'):
                    #break

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
