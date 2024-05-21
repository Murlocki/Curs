
from view import settings
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from model import YoloDetect
from model import ComplexModel
import os
from pathlib import Path
import validators
class MainWindowController():
    def __init__(self,window):
        self._window = window
        self._model = ComplexModel.ComplexModel()
        self._model.create_model()
    def clicked_settings(self):
        sets = settings.Settings(self.model)
        sets = sets.create()
        sets.show()

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



    def clicked_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(self.window,'Open file','/home')
        self.window.Input.setText(dir)
    def clicked_file(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(self.window,'Open file','/home')[0]
        self.window.Input.setText(fname)

    def clicked_start(self):
        f = Path(self.window.Input.text())
        if (f.is_file() or f.is_dir() ) and self.window.Input.text() != '':
            show_regime = 1 if self.window.img.isChecked() else 2

            #Выбираем модель классификации
            classify_model="inception"
            if self.window.dense_net.isChecked():
                classify_model = "dense"
            elif self.window.inception.isChecked():
                classify_model = "inception"
            elif self.window.yolo.isChecked():
                classify_model = "yolo"

            self.model.create_model(classify_model)
            self.model.process(self.window.Input.text(),show_regime)

        else:
            m = QMessageBox(1, "Выбор файлов", "Неверно выбрано значение")
            m.setStyleSheet("background-color:white")
            m.exec_()

    def clicked_clear(self):
        with open(os.path.join(os.path.dirname(__file__), '..\logs\logs.txt'), 'w') as file:
            pass
        m = QMessageBox(1, "Очистка логов", "Успешное очищено")
        m.setStyleSheet("background-color:white")
        m.exec_()

