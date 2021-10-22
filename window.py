import numpy as np
import cv2
import scipy.signal as signal
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QGridLayout, QPushButton, QProgressBar, QStyle
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5 import FigureCanvasQT
from threading import Thread
from queue import Queue, Empty
from time import time
from app import *
from QLabeledSpinBox import *
from QLabeledProgressBar import *

DEFAULT_FS = 30

class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    runs = True
    
    def __init__(self, App, Fs):
        super().__init__()
        self.App = App
        self.Fs = Fs

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap = cv2.VideoCapture('videos/breathing_12bpm.mp4')
        n = 0
        while cap.isOpened() and self.runs:
            ret, frame = cap.read()
            n += 1
            if n % (DEFAULT_FS//self.Fs) != 0:
                continue
            
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                try:
                    self.App.new_sample(frame)
                    x, y, width, height = self.App.bbox
                    for (x_bb, y_bb, w_bb, h_bb) in self.App.rois:
                        cv2.rectangle(frame, (int(x), int(y)), (int(x+width), int(y+height)), (0, 255, 0), 2, 1)
                        cv2.rectangle(frame, (x_bb, y_bb), (x_bb+w_bb, y_bb+h_bb), (0, 255, 0), 2)
                        
                    frameRect =  cv2.flip(frame, 1)

                except SampleError as err:
                    frameRect =  cv2.flip(frame, 1)
                    
                except:
                    pass
                
                if self.App.HeartRateValid:
                    cv2.putText(frameRect, "Heart Rate: {:.1f} bpm".format(self.App.HeartRate), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
                else:
                    cv2.putText(frameRect, "Heart Rate: {:.1f} bpm".format(self.App.HeartRate), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                
                cv2.putText(frameRect, "Breathing Rate: {:.1f} bpm".format(self.App.RespRate), (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)     
                
                if len(self.App.brightness[0]) != 0:
                    # display brigness and ratio
                    cv2.putText(frameRect, "brightness: {}".format([int(num[((len(num)-1)//30)*30]) for num in self.App.brightness]), (40,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    cv2.putText(frameRect, "distance ratio: {}".format([int(10000*num[((len(num)-1)//30)*30]) for num in self.App.distance_ratio]), (40,160), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
                 
                cv2.putText(frameRect, "snr: {:.1f}".format(self.App.snr[-1]), (40,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)     
                
                rgbImage = cv2.cvtColor(frameRect, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                
            else:
                print('Video off')
                break
                
        cap.release()
               
    def quit(self):
        self.runs = False
        
    
class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
    
        # location and size of window that opens
        self.title = 'heart-rate'
        self.left = 100
        self.top = 100
        self.width = 1500
        self.height = 800
        self.Fs = 30
        self.n_seconds = 20
        self.t = np.linspace(start=0, stop=self.n_seconds, num=self.n_seconds*self.Fs, endpoint=False)
        self.t_rri = np.linspace(start=0, stop=2*self.n_seconds, num=2*self.n_seconds*self.Fs, endpoint=False)
        self.newData = None
        
        self.initUI()

    # @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        # print('frame displayed')
        
    def WelchUpdate(self, n):
        WelchLine, = self.WelchAx.get_lines()
        
        try:
            f, pxx = self.App.WelchQueue.get_nowait()
            WelchLine.set_data(f*60, pxx)
            # self.WelchAx.set_ylim([0, pxx.max()])
            
        except Empty:
            pass
        
        return WelchLine, 
        
    def RespUpdate(self, n):
        
        ppgLine, maxLine = self.ppgAx.get_lines()
        rriLine, = self.rriAx.get_lines()
        lombLine, = self.lombAx.get_lines()
    
        
        try:
            filtered_signal = self.App.SignalQueue.get_nowait()
      
            ppgLine.set_data(self.t[:filtered_signal.shape[0]], filtered_signal[-self.n_seconds*self.App.Fs:])
            try:
                self.ppgAx.set_ylim([filtered_signal[-self.n_seconds*self.App.Fs:].min(), filtered_signal[-self.n_seconds*self.App.Fs:].max()])
            except ValueError:
                print('max:', filtered_signal[-self.n_seconds*self.App.Fs:].max())
                print('min:', filtered_signal[-self.n_seconds*self.App.Fs:].min())
                # print(filtered_signal)
                
            try:
                newData = self.App.RespQueue.get_nowait()
                self.newData = newData
            
            except Empty:
                if self.newData is not None:
                    newData = self.newData
                else:
                    return ppgLine, maxLine, rriLine, lombLine
                
            
            shift_indx = max(0, filtered_signal.shape[0]-self.n_seconds*self.App.Fs)
            shift_indx_rri = max(0, filtered_signal.shape[0]-2*self.n_seconds*self.App.Fs)
            peak_times = newData['peak_times'][newData['peak_times'] >= shift_indx]
            peak_times_rri = newData['peak_times'][newData['peak_times'] >= shift_indx_rri]
            rri = newData['rri'][-len(peak_times_rri):]/self.App.Fs
            
            maxLine.set_data(self.t[peak_times-shift_indx], filtered_signal[peak_times])
            rriLine.set_data(self.t_rri[peak_times_rri-shift_indx_rri], rri)
            lombLine.set_data(60*newData['freqs'], newData['pgram'])
            
            # self.rriAx.set_ylim([rri.min(), rri.max()])
            # self.lombAx.set_ylim([0, newData['pgram'].max()])
            
        except Empty:
            pass
        
        return ppgLine, maxLine, rriLine, lombLine
    
    def closeEvent(self, event):
        # self.VideoSource.quit()
        # self.VideoSource.wait()
        event.accept()
    

    def initUI(self):
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # self.resize(640, 480)
        
        # create a label for live video
        self.label = QLabel(self)
        self.grid_layout.addWidget(self.label, 0, 0, 1, 1) # row, col, hight, width
        self.label.setAlignment(Qt.AlignCenter)
        self.label.resize(640, 480)
        
        
        # organize buttons in grid
        self.buttons_widget = QWidget()
        self.buttons_grid = QGridLayout()
        self.buttons_widget.setLayout(self.buttons_grid)
        self.grid_layout.addWidget(self.buttons_widget, 1, 0, 2, 1)
        
        # organize ProgressBars in grid
        self.progressbars_widget = QWidget()
        self.progressbars_grid = QGridLayout()
        self.progressbars_widget.setLayout(self.progressbars_grid)
        self.buttons_grid.addWidget(self.progressbars_widget, 0, 0, 1, 2)
        
        # add spinbox for welch history
        self.welchSpinBox = QLabeledSpinBox(label='welch no.\nof windows')
        self.welchSpinBox.setGeometry(0, 0, 20, 20)
        self.buttons_grid.addWidget(self.welchSpinBox, 1, 1, 1, 1)
        
        # add spinbox for resp history
        self.respSpinBox = QLabeledSpinBox(label='welch\nwindow length')
        self.respSpinBox.setGeometry(0, 0, 20, 20)
        self.buttons_grid.addWidget(self.respSpinBox, 1, 2, 1, 1)
        
        # add spinbox for welch window size
        self.welchWinSizeSpinBox = QLabeledSpinBox(label='lomb no.\nof windows')
        self.welchWinSizeSpinBox.setGeometry(0, 0, 20, 20)
        self.buttons_grid.addWidget(self.welchWinSizeSpinBox, 1, 3, 1, 1)
        
        # add reset button
        self.resetButton = QPushButton('reset')
        self.resetButton.setGeometry(20, 20, 40, 40)
        self.buttons_grid.addWidget(self.resetButton, 1, 0, 1, 1)
        
        # progress bar
        self.snrLevelBar = QLabeledProgressBar(objectName='SNR', textVisible=True, label='snr')
        self.progressbars_grid.addWidget(self.snrLevelBar, 0, 0, 1, 1)
        
        # progress bar
        self.brightnessLevel = QLabeledProgressBar(objectName='SNR', textVisible=True, label='light')
        self.progressbars_grid.addWidget(self.brightnessLevel, 0, 1, 1, 1)
        
        # progress bar
        self.distanceLevel = QLabeledProgressBar(objectName='SNR', textVisible=True, label='distance')
        self.progressbars_grid.addWidget(self.distanceLevel, 0, 2, 1, 1)
        
        # Label for hr and rr data
        self.HrLabel = QLabel()
        self.buttons_grid.addWidget(self.HrLabel, 0, 2, 1, 2) # row, col, hight, width
        self.HrLabel.setAlignment(Qt.AlignCenter)
        self.HrLabel.setStyleSheet("""QLabel { 
                                   background-color : white; 
                                   color : green;
                                   font-size : 12pt; 
                                   }""")
        self.HrLabel.setText('Heart-Rate: 65 [bpm]\n\nBreathing-Rate: 12 [bpm]')
        
        # # add figure for welch periodogram
        # self.WelchFig = Figure(figsize=(7,2)) # width, hight
        # self.WelchCanvas = FigureCanvas(self.WelchFig)
        # self.grid_layout.addWidget(self.WelchCanvas, 2, 0, 1, 2)
        
        # add figure for respiratory rate
        self.RespFig = Figure(figsize=(7, 9)) #(4,9)
        self.RespCanvas = FigureCanvas(self.RespFig)
        self.grid_layout.addWidget(self.RespCanvas, 0, 2, 3, 2)
        
        gs = self.RespFig.add_gridspec(3,2)
        self.hrAx = self.RespFig.add_subplot(gs[0, :])
        self.ppgAx = self.RespFig.add_subplot(gs[1, :])
        self.WelchAx = self.RespFig.add_subplot(gs[2, 0])
        self.lombAx = self.RespFig.add_subplot(gs[2, 1])
        

        self.WelchAx.plot([], [])
        self.WelchAx.set_xlabel('bpm')
        self.WelchAx.set_title('welch periodogram')
        self.WelchAx.set_xlim([0, 180])
        self.WelchAx.set_ylim([0, 1.1])
        
        
        # self.ppgAx, self.rriAx, self.lombAx = self.RespFig.subplots(nrows=3, ncols=2)
        self.ppgAx.plot([],[])
        self.ppgAx.plot([], [], "x")
        self.ppgAx.set_xlim([0, self.n_seconds])
        self.ppgAx.set_xlabel('time')
        self.ppgAx.set_title('ppg signal')

        self.hrAx.plot([], [])
        self.hrAx.set_xlim([0, 2*self.n_seconds])
        self.hrAx.set_ylim([0.5, 1.7])
        self.hrAx.set_xlabel('time')
        self.hrAx.set_ylabel('hr [bpm]')
        self.hrAx.set_title('heart-rate signal')
        
        self.lombAx.plot([], [])
        self.lombAx.set_xlim([0, 40])
        self.lombAx.set_ylim([0, 1.1])
        self.lombAx.set_xlabel('breaths per minute')
        self.lombAx.set_title('lomb periogogram')
        
        # self.WelchFig.tight_layout()
        self.RespFig.tight_layout()
        
        self.App = App(Fs=self.Fs)
        
        # self.Welchani = FuncAnimation(self.RespFig, self.WelchUpdate, blit=True, interval=100) 
        # self.RespAni = FuncAnimation(self.RespFig, self.RespUpdate, blit=True, interval=100)
        
        self.VideoSource = VideoThread(self.App, Fs=self.Fs)
        self.VideoSource.changePixmap.connect(self.setImage)
        self.VideoSource.finished.connect(self.close)
        self.VideoSource.start()
    
    
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    sys.exit(app.exec_())