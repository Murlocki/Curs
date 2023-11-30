
from view import settings
from PyQt5.QtWidgets import QFileDialog
class MainWindowController():
    def __init__(self,window):
        self.window = window
        self.settings = None
    def clicked_settings(self):
        self.settings = settings.Settings()
        self.settings = self.settings.create()
        self.settings.show()

    def clicked_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(self.window,'Open file','/home')
        self.window.Input.setText(dir)
    def clicked_file(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(self.window,'Open file','/home')[0]
        self.window.Input.setText(fname)

    def clicked_start(self):
        pass
    def close(self):
        if self.settings is not None:
            self.settings.close()
        self.window.close()
