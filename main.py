from view.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
if __name__ == '__main__':
    #import webbrowser
    #path = r"C:\Users\kiril\PycharmProjects\Curs\images"
    #webbrowser.open(path)

    app = QApplication(sys.argv)
    w = MainWindow().create()
    w=w.show()
    sys.exit(app.exec_())
