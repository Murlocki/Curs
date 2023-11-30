import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from view import settings
from controller import main_window_controller
from images import images_paths
import os
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = main_window_controller.MainWindowController(self)

    def create(self):
        uic.loadUi(os.path.join(os.path.dirname(__file__), '..\\ui\\main.ui'), self)

        # Ивент открытия
        self.settings.clicked.connect(lambda : self.controller.clicked_settings())
        self.dir.clicked.connect(lambda :self.controller.clicked_dir())
        self.file.clicked.connect(lambda: self.controller.clicked_file())
        self.start.clicked.connect(lambda :self.controller.clicked_start())

        return self
    def show(self):
        super().show()
        return self

    def closeEvent(self, QCloseEvent):
        self.controller.close()




