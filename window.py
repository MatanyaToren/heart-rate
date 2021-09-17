import numpy as np
import cv2
import scipy.signal as signal
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5 import FigureCanvasQT
from time import time
from tracking import *


class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    runs = True
    tracker = FaceTracker()
    
    def __init__(self, Fs=30, min_bpm=45, max_bpm=200):
        super().__init__()
        
        self.min_bpm, self.max_bpm = min_bpm, max_bpm
        self.offset_x = 0.1
        self.offset_y = 0.6
        self.size_x = 0.2
        self.size_y = 0.2

        self.raw_signal = []
        self.filtered_signal = []
        
        self.bandPass = signal.firwin(100, np.array([min_bpm, max_bpm])/60, fs=Fs, pass_zero=False)
        self.z = np.zeros(self.bandPass.shape[-1]-1)
        

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self.runs:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                try:
                    x, y, width, height = self.tracker.update(frame)
                    cv2.rectangle(frame, (int(x), int(y)), (int(x+width), int(y+height)), (0, 255, 0), 2, 1)
                    frameRect =  cv2.flip(frame, 1)
                    
                    self.get_signal(frame, (x, y, width, height))

                except (TrackingError, OutOfFrameError, DetectionError) as err:
                    frameRect =  cv2.flip(frame, 1)
                     
                rgbImage = cv2.cvtColor(frameRect, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                
        cap.release()
                
    def quit(self):
        self.runs = False
        
    def get_signal(self, frame, bbox):
        """
        get signal from roi
        """
        x, y, w, h = bbox
        
        x_bb = int(x + self.offset_x * w)
        w_bb = int(w * self.size_x)
        y_bb = int(y + self.offset_y * h)
        h_bb = int(h * self.size_y)
        
        self.roi = (x_bb, y_bb, w_bb, h_bb)
        
        # spatial mean of the bounding box of the face
        self.raw_signal.append(np.mean(frame[x_bb:x_bb+w_bb, y_bb:y_bb+w_bb, 1][:]))

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        # location and size of window that opens
        self.title = 'heart-rate'
        self.left = 100
        self.top = 100
        self.width = 1500
        self.height = 800
        
        self.initUI()

    # @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        # print('frame displayed')
        
    
    
    def closeEvent(self, event):
        self.app.quit()
        self.app.wait()
        event.accept()
        print('closed window')

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
        self.WelchFig = Figure(figsize=(2,1)) # width, hight
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
        self.ppgAx.plot([],[])
        self.ppgAx.plot([], [], "x")
        self.ppgAx.set_xlabel('sample')

        self.rriAx.plot([], [])
        self.rriAx.set_xlabel('sample time')
        self.rriAx.set_ylabel('rri')
        
        self.lombAx.plot([], [])
        self.lombAx.set_xlabel('theta [rad]')
        
        self.RespFig.tight_layout()
        
        self.app = VideoThread()
        self.app.changePixmap.connect(self.setImage)
        self.app.start()
    
    
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    sys.exit(app.exec_())