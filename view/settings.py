from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5 import uic
from images import  images_paths
import os
from controller import settings_controller
class Settings(QMainWindow):
    def __init__(self,model):
        super().__init__()
        self.controller = settings_controller.SettingsController(self,model)

    def create(self):
        uic.loadUi(os.path.join(os.path.dirname(__file__), '..\\ui\\settings.ui'), self)

        self.weight_but.clicked.connect(lambda :self.controller.choose_weights())
        self.cancel.clicked.connect(lambda : self.controller.close())
        self.accept.clicked.connect(lambda: self.controller.change_model())

        self.weights.setText(self.controller.model.path_weights)
        self.conf.setText(str(self.controller.model.conf))
        self.iou.setText(str(self.controller.model.iou))
        self.imgsz.setText(str(self.controller.model.imgsz))
        self.class_conf.setText(str(self.controller.model.classify_conf))
        self.bright_coef.setText(str(self.controller.model.bright_coef))
        self.contr_coef.setText(str(self.controller.model.contrast_coef))
        self.color_coef.setText(str(self.controller.model.color_coef))
        return self

    def show(self):
        super().show()
        return self