from view import settings
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

from pathlib import Path

from model import yoloModel


class SettingsController():
    def __init__(self, window, model):
        self.window = window
        self.model = model

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
        if k == 4:
            conf = float(conf)
            iou = float(iou)
            imgsz = int(imgsz)
            self.model.change_parameters(imgsz, w_path, conf, iou)
            self.close()

    def create_model(self):
        self.model = yoloModel.YoloModel()

    def close(self):
        self.window.close()
