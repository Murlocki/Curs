from abc import ABC,abstractmethod

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from matplotlib import pyplot as plt
from overrides import override


class ProccessModel(ABC):
    def __init__(self,detectModel,classifyModel,processPath,savePath):
        self._detectModel = detectModel
        self._classifyModel = classifyModel
        self._processPath = processPath
        self._savePath = savePath

    @property
    def detectModel(self):
        return self._detectModel
    @detectModel.setter
    def detectModel(self,new):
        self._detectModel = new

    @property
    def classifyModel(self):
        return self._classifyModel

    @classifyModel.setter
    def classifyModel(self, new):
        self._classifyModel = new

    @property
    def processPath(self):
        return self._processPath

    @processPath.setter
    def processPath(self, new):
        self._processPath = new

    @property
    def savePath(self):
        return self._savePath

    @savePath.setter
    def savePath(self, new):
        self._savePath = new

    @abstractmethod
    def process(self):
        pass


class VideoModel(ProccessModel):
    @override
    def process(self):
        cap = cv2.VideoCapture(self.processPath)
        ret = True
        time_skips = float(150)
        count = 0
        while ret:
            ret, frame = cap.read()
            if ret:
                results = self.detectModel.process(frame, self.savePath)
                results = self.processOne(results,frame)
                cap.set(cv2.CAP_PROP_POS_MSEC,(count * time_skips))
                count+=1
                cv2.imshow('frame',results)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
    def processOne(self,results,original_frame):
        for result in results:
            boxes = result.boxes.cpu().numpy().xyxyn
            dh, dw, _ = original_frame.shape
            boxes_draw = []
            for box in boxes:
                x1, y1, x2, y2 = int(box[0] * dw), int(box[1] * dh), int(box[2] * dw), int(box[3] * dh)
                img_crop = original_frame[y1:y2, x1:x2]
                PIL_image = Image.fromarray(np.uint8(img_crop)).convert('RGB')

                probs_values, probs_indeces = self.classifyModel.process(PIL_image)
                print(probs_values)
                print(probs_indeces)
                if (probs_values[0][np.where(probs_indeces[0] == 0)] >= self.classifyModel.classify_conf
                        or probs_values[0][np.where(probs_indeces[0] == 1)] >= self.classifyModel.classify_conf):
                    boxes_draw.append([x1, y1, x2, y2, probs_indeces[0][0]])
            for box in boxes_draw:
                # Draw a rectangle
                color = (0, 255, 0) if box[4] == 0 else (0, 0, 255)
                cv2.rectangle(original_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        return original_frame

class DirModel(ProccessModel):

    @override
    def process(self):
        results = self.detectModel.process(self.processPath,self.savePath)
        print(self.detectModel.model.names)
        number = 1
        for result in results:
            img = cv2.imread(result.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = result.boxes.cpu().numpy().xyxyn
            dh,dw,_ = img.shape
            boxes_draw = []
            k = 0
            cls = result.boxes.cls.cpu().numpy()
            for box in boxes:
                if self.detectModel.model.names[cls[k]]!="shopping-trolley":
                    continue
                k=k+1

                x1, y1, x2, y2 = int(box[0] * dw), int(box[1] * dh), int(box[2] * dw), int(box[3] * dh)
                img_crop = img[y1:y2,x1:x2]
                PIL_image = Image.fromarray(np.uint8(img_crop)).convert('RGB')

                probs_values,probs_indeces = self.classifyModel.process(PIL_image)
                print(probs_values)
                print(probs_indeces)
                text = "leura:" + str(probs_values[0][np.where(probs_indeces[0] == 0)]) + "\nmagnit:" + str(
                    probs_values[0][np.where(probs_indeces[0] == 1)]) + "\nother:" + \
                       str(probs_values[0][np.where(probs_indeces[0]==2)])
                draw = ImageDraw.Draw(PIL_image)
                font = ImageFont.truetype('arial.ttf', 24)
                draw.text((10, 10), text, font=font, fill=(255, 255, 0))
                PIL_image.save(self.savePath+'\\classify\\'+str(number)+'.jpg')
                number+=1

                if(probs_values[0][np.where(probs_indeces[0] == 0)]>=self.classifyModel.classify_conf
                    or probs_values[0][np.where(probs_indeces[0] == 1)]>=self.classifyModel.classify_conf):
                    boxes_draw.append([x1,y1,x2,y2,probs_indeces[0][0]])

            image = cv2.imread(result.path)
            for box in boxes_draw:
                # Draw a rectangle
                color = (0, 255, 0) if box[4]==0 else (0,0,255)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.imwrite(self.savePath+"\\result\\"+result.path.split("\\")[-1],image)