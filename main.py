from view.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow().create()
    w.show()
    sys.exit(app.exec_())
