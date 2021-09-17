import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QVBoxLayout, QGridLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5 import FigureCanvasQT
from time import time
from App import *

# class VideoThread(QThread):
#     changePixmap = pyqtSignal(QImage)
#     runs = True

#     def run(self):
#         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         while self.runs:
#             ret, frame = cap.read()
#             if ret:
#                 # https://stackoverflow.com/a/55468544/6622587
#                 start = time()
#                 rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 h, w, ch = rgbImage.shape
#                 bytesPerLine = ch * w
#                 convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
#                 p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
#                 print('diff {:.4f}'.format(time()-start))
#                 self.changePixmap.emit(p)
                
#         cap.release()
                
#     def quit(self):
#         self.runs = False

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
        
    def WelchUpdate(self, iter):
        """
        updates welch figure with new data
        """
        WelchLine, = self.WelchAx.get_lines()
        
        if self.app.welch_data_available.is_set():
            WelchLine.set_data(self.app.f, self.app.pxx)
            # self.WelchAx.relim()
            # update ax.viewLim using the new dataLim
            # self.WelchAx.autoscale_view()
            # self.WelchFig.canvas.draw_idle()
        
        return WelchLine, 
        
    
    def RespUpdate(self, iter):
        """
        updates repiratory figure with new data
        """
        
        # get lines
        ppgLine, ppgMaxLine = self.ppgAx.get_lines()
        rriLine, = self.rriAx.get_lines()
        lombLine, = self.lombAx.get_lines()
        
        if self.app.resp_data_available.is_set():
            # set data
            time = np.arange(self.app.n+1)/self.app.Fs
            ppgLine.set_data(time, self.app.filtered_signal)
            ppgMaxLine.set_data(time[np.array(self.app.resp.peak_times)], np.array(self.app.filtered_signal)[np.array(self.app.resp.peak_times)])
            rriLine.set_data(np.array(self.app.resp.peak_times)/self.app.Fs, np.array(self.app.resp.rri)/self.app.Fs)
            lombLine.set_data(self.app.resp.freqs, self.app.resp.pgram)
            
            self.ppgAx.set_xlim([time[-1]-10, time[-1]])
            self.rriAx.set_xlim([time[-1]-10, time[-1]])
            
            # update ax.viewLim using the new dataLim
            # self.ppgAx.autoscale_view()
            # self.rriAx.autoscale_view()
            # self.lombAx.autoscale_view()
            
            # draw changes
            # self.RespFig.canvas.draw_idle()
        
        return ppgLine, ppgMaxLine, rriLine, lombLine

    
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
        
        self.app = App()
        self.app.changePixmap.connect(self.setImage)
        self.app.start()
        
        # create animation for figures
        self.WelchAnim = FuncAnimation(fig=self.WelchFig, func=self.WelchUpdate, blit=True, interval=200)
        self.RespAnim = FuncAnimation(fig=self.RespFig, func=self.RespUpdate, blit=True, interval=200)
        
        
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    sys.exit(app.exec_())