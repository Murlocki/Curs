from abc import ABC, abstractmethod
import numpy
import numpy as np
import torch
import torchvision
from overrides import override
from torch.autograd import Variable
from torchvision import transforms
from ultralytics import YOLO


#Абстрактный класс для модели классификации
class ClassifyModel(ABC,object):
    def __init__(self,weight_path,classify_conf):
        self._weight_path = weight_path
        self._classify_conf = classify_conf
        self._model = None
        self._device = None
        self._transform = None

    #Геттеры и сеттеры модели
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,new):
        self._model = new

    #Геттеры и сеттеры предобработки изображений
    @property
    def transform(self):
        return self.transform
    @transform.setter
    def transform(self,new):
        self._transform=new

    #Геттер и сеттер устройства обработки
    @property
    def device(self):
        return self._device
    @device.setter
    def device(self,new):
        self._device = new

    #Метод инициализации модели
    @abstractmethod
    def createModel(self):
        pass

    #Геттеры и сеттеры пути до весов
    @property
    def weight_path(self):
        return self._weight_path
    @weight_path.setter
    def weight_path(self,new):
        self._weight_path = new

    #Геттеры и сеттеры точности классификации
    @property
    def classify_conf(self):
        return self._classify_conf
    @classify_conf.setter
    def classify_conf(self,new):
        self._classify_conf = new

    #Метод обработки 1 изображения
    def proccess(self,pil_image)->(numpy.array,numpy.array):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            data = self.transform(pil_image)
            input_batch = data.unsqueeze(0)
            input_batch = Variable(input_batch.to(self.device))
            output = self.model(input_batch)
            print(output)
            probs = torch.softmax(output, dim=1)
            print(probs)
            top_k_probs, top_k_indices = probs.topk(3, dim=1)
            print(top_k_probs)
            print(top_k_indices)

            probs_values = top_k_probs.cpu().detach().numpy()
            probs_indeces = top_k_indices.cpu().detach().numpy()

            return probs_values, probs_indeces

    #Метод вывода логов
    @abstractmethod
    def write_log(self,results):
        pass

class DenseModel(ClassifyModel):
    @override
    def createModel(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = torchvision.models.densenet121()
        self.model.load_state_dict(
            torch.load(self.weight_path))

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class InceptionModel(ClassifyModel):
    @override
    def createModel(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model= torchvision.models.inception_v3()
        self.model.load_state_dict(
            torch.load(self.weight_path))

        self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
class YoloModel(ClassifyModel):
    @override
    def createModel(self):
        self.model = YOLO(self.weight_path)
    @override
    def proccess(self,pil_image)->(numpy.array,numpy.array):
        results = self.model(pil_image)
        probs_values = [results[0].probs.top5conf.cpu().detach().numpy()]
        probs_indeces = np.array([results[0].probs.top5])
        print(probs_values)
        print(probs_indeces)

        return probs_values,probs_indeces
