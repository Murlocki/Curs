import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from view.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
from ultralytics import YOLO
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow().create()
    w=w.show()
    sys.exit(app.exec_())

