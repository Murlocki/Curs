
from view import settings
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from model import yoloModel
import os
from pathlib import Path
import validators
class MainWindowController():
    def __init__(self,window):
        self._window = window
        self._model = yoloModel.YoloModel()
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
            if self.window.img.isChecked():
                self.model = self.model.create_model(self.window.Input.text(), 1)
                print(1)
            elif self.window.video.isChecked():
                self.model = self.model.create_model(self.window.Input.text(), 2)
                print(2)
            self.model.process(self.window.Input.text())

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

