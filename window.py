import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5 import FigureCanvasQT

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        # location and size of window that opens
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 1500
        self.height = 800
        self.initUI()

    # @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # self.resize(640, 480)
        
        # create a label for live video
        self.label = QLabel(self)
        self.grid_layout.addWidget(self.label, 0, 0, 2, 2) # row, col, hight, width
        self.label.setAlignment(Qt.AlignCenter)
        self.label.resize(640, 480)
        
        # add figure for welch periodogram
        self.WelchFig = Figure(figsize=(3,2)) # width, hight
        self.WelchCanvas = FigureCanvas(self.WelchFig)
        self.grid_layout.addWidget(self.WelchCanvas, 2, 0, 1, 2)
        
        # add figure for respiratory rate
        self.RespFig = Figure(figsize=(4,9))
        self.RespCanvas = FigureCanvas(self.RespFig)
        self.grid_layout.addWidget(self.RespCanvas, 0, 2, 3, 2)
        
        
        # set up figures
        self.WelchAx = self.WelchFig.subplots()
        self.WelchAx.plot([], [])
        self.WelchAx.set_xlabel('bpm')
        self.WelchAx.set_title('welch periodogram')
        self.WelchAx.set_xlim([0, 180])
        
        self.ppgAx, self.rriAx, self.lombAx = self.RespFig.subplots(nrows=3, ncols=1)
        self.ppgAx.plot([], [])
        self.ppgAx.plot([], [], "x")
        self.ppgAx.set_xlabel('sample')

        self.rriAx.plot([], [])
        self.rriAx.set_xlabel('sample time')
        self.rriAx.set_ylabel('rri')
        
        self.lombAx.plot([], [])
        self.lombAx.set_xlabel('theta [rad]')
        
        self.RespFig.tight_layout()
        
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        

        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())