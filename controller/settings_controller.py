from view import settings
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

from pathlib import Path

from model import YoloDetect


class SettingsController():
    def __init__(self, window, model):
        self._window = window
        self._model = model

    @property
    def window(self):
        return self._window
    @window.setter
    def window(self,new):
        self._window = new

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,new):
        self._model = new


    def choose_weights(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(self.window, 'Open file', '/home','pt(*.pt)')[0]
        self.window.weights.setText(fname)

    def change_model(self):

        k = 0

        w_path = self.window.weights.text()

        file = Path(w_path)
        if not file.is_file() and w_path!='':
            m = QMessageBox(1, "Выбор весов", "Неверно выбрано значение")
            m.setStyleSheet("background-color:white")
            m.exec_()
        else:
            k = k+1

        if k == 1:
            conf = self.window.conf.text()
            try:
                if float(conf) >= -1:
                    k = k+1
            except Exception:
                m = QMessageBox(1, "Выбор минмального значения точности", "Неверно выбрано значение",)
                m.setStyleSheet("background-color:white")
                m.exec_()

        if k == 2:
            iou = self.window.iou.text()
            try:
                if float(iou) >= -1:
                    k = k+1
            except Exception:
                m = QMessageBox(1, "Выбор значения наложения", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()

        if k == 3:
            imgsz = self.window.imgsz.text()
            try:
                if int(imgsz) >= -1:
                    k = k+1
            except Exception:
                m = QMessageBox(1, "Выбор разрешения", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()
        if k==4:
            classify_conf = self.window.class_conf.text()
            try:
                if float(classify_conf)>=0:
                    k=k+1
            except Exception:
                m = QMessageBox(1, "Выбор точности классификации", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()
        if k==5:
            color_coef = self.window.color_coef.text()
            try:
                if float(color_coef)>0:
                    k=k+1
            except Exception:
                m = QMessageBox(1, "Выбор коэффициента увеличения насыщенности изображения", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()
        if k==6:
            bright_coef = self.window.bright_coef.text()
            try:
                if float(bright_coef)>=0:
                    k=k+1
            except Exception:
                m = QMessageBox(1, "Выбор коэффициента яркости изображения", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()
        if k==7:
            contrast_coef = self.window.contr_coef.text()
            try:
                if float(contrast_coef)>=0:
                    k=k+1
            except Exception:
                m = QMessageBox(1, "Выбор коэффициента контраста изображения", "Неверно выбрано значение")
                m.setStyleSheet("background-color:white")
                m.exec_()
        if k == 8:
            conf = float(conf)
            iou = float(iou)
            imgsz = int(imgsz)
            classify_conf = float(classify_conf)
            bright_coef = float(bright_coef)
            contrast_coef = float(contrast_coef)
            color_coef = float(color_coef)
            self.model.change_parameters(imgsz, w_path, conf, iou,classify_conf,bright_coef,contrast_coef,color_coef)
            self.close()




    def close(self):
        self.window.close()
