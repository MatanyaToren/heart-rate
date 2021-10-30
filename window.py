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

def runEmpty(cap):
    for __ in range(5):
        ret, frame = cap.read()


class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeSnr = pyqtSignal(float)
    changeLight = pyqtSignal(float)
    changeDistance = pyqtSignal(float)
    changeMovement = pyqtSignal(float)
    changeHrResp = pyqtSignal(dict)
    
    
    runs = True
    
    def __init__(self, App, Fs):
        super().__init__()
        self.App = App
        self.Fs = Fs

    def run(self):
        rgbImage = np.zeros((480,640,3), dtype=np.uint8)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        
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
                    
                    

                    if (n % self.Fs) == 0 and len(self.App.brightness[0]) != 0:
                        self.changeSnr.emit(self.App.snr[-1])
                        self.changeLight.emit(np.mean([level[-1] for level in self.App.brightness]))
                        self.changeDistance.emit(np.mean([level[-1] for level in self.App.distance_ratio]))
                        self.changeMovement.emit(self.App.movement_indicator[-1])
                        self.changeHrResp.emit({'hr': self.App.HeartRate[-1], 
                                                'hrValid': self.App.HeartRateValid[-1], 
                                                'resp': self.App.RespRate[-1],
                                                'respValid': self.App.RespRateValid[-1]})
                    
                except SampleError as err:
                    frameRect =  cv2.flip(frame, 1)
                    
                except Exception as err:
                    print(err)
                    frameRect =  cv2.flip(frame, 1)
                    
                except:
                    print('unknown error while using new frame')
                
        
                try:    
                    rgbImage = cv2.cvtColor(frameRect, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
                
                except:
                    print('an error ocurred while sending frame to display')
                
            else:
                print('Video off')
                break
        # print('capture object has closed')        
        cap.release()
               
    def quit(self):
        self.runs = False
        
    
class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
    
        # location and size of window that opens
        self.title = 'heart-rate'
        self.left = 40
        self.top = 80
        self.width = 1500
        self.height = 900
        self.Fs = 30
        self.n_seconds = 20
        self.t = np.linspace(start=0, stop=self.n_seconds, num=self.n_seconds*self.Fs, endpoint=False)
        self.t_rri = np.linspace(start=0, stop=2*self.n_seconds, num=2*self.n_seconds*self.Fs, endpoint=False)
        self.newData = None
        
        # progress bar messages
        self.messagesToUser = {'brightness': '- Please move so that your<br> face will be under direct light',
                               'distance': '- Please get closer<br> to the camera',
                               'movement': '- Please try to stay still'}
        self.messagesToUser_On = {'brightness': False, 
                                  'distance': False,
                                  'movement': False}
        
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
        # rriLine, = self.rriAx.get_lines()
        lombLine, = self.lombAx.get_lines()
        hrLine, = self.hrAx.get_lines()
        respLine, = self.respAx.get_lines()
    
        WelchLine, = self.WelchAx.get_lines()
        
        try:
            WelchData = self.App.WelchQueue.get_nowait()
            WelchLine.set_data(WelchData['f']*60, WelchData['pxx'])
            
            offset_hr = WelchData['HeartRateTime'][-1] - 2*self.n_seconds if WelchData['HeartRateTime'][-1] > 2*self.n_seconds else 0
            hrLine.set_data(WelchData['HeartRateTime'] - offset_hr,
                            WelchData['HeartRate'])
            self.WelchAx.set_ylim([0, max(np.nanmax(WelchData['pxx']), 1.1)])
            
        except Empty:
            pass
        
        except BaseException as e:
            print(e)
            
        except:
            print('unknown exception at RespUpdate')
         
        
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
                
                offset_rr = newData['RespRateTime'][-1] - 2*self.n_seconds if newData['RespRateTime'][-1] > 2*self.n_seconds else 0
                respLine.set_data(newData['RespRateTime'] - offset_rr,
                            newData['RespRate'])
            
            except Empty:
                if self.newData is not None:
                    newData = self.newData
                else:
                    return ppgLine, maxLine, lombLine, respLine, hrLine, WelchLine
                
            
            shift_indx = max(0, filtered_signal.shape[0]-self.n_seconds*self.App.Fs)
            # shift_indx_rri = max(0, filtered_signal.shape[0]-2*self.n_seconds*self.App.Fs)
            peak_times = newData['peak_times'][newData['peak_times'] >= shift_indx]
            # peak_times_rri = newData['peak_times'][newData['peak_times'] >= shift_indx_rri]
            # rri = newData['rri'][-len(peak_times_rri):]/self.App.Fs
            
            maxLine.set_data(self.t[peak_times-shift_indx], filtered_signal[peak_times])
            # rriLine.set_data(self.t_rri[peak_times_rri-shift_indx_rri], rri)
            lombLine.set_data(60*newData['freqs'], newData['pgram'])
            
            # self.rriAx.set_ylim([rri.min(), rri.max()])
            self.lombAx.set_ylim([0, max(newData['pgram'].max(), 1.1)])
            
        except Empty:
            pass
        
        return ppgLine, maxLine, lombLine, respLine, hrLine, WelchLine
    
    def updateVitalsDisplay(self, result: dict = {'hr': 65, 'hrValid': False, 'resp': 12, 'respValid': False}):
        result['hrColor'] = 'green' if result['hrValid'] is True else 'red'
        result['respColor'] = 'green' if result['respValid'] is True else 'red'
        
        self.HrLabel.setText(('<font color="{hrColor}"> {hr:.0f}</font>'
                              + '<br><font color="black">[bpm]</font>').format(**result))

        self.RespLabel.setText(('<font color="{respColor}"> {resp:.0f}</font>'
                              + '<br><font color="black">[bpm]</font>').format(**result))
        
        # self.RespLabel.setText(('<br><font color="black">&nbsp; Heart Rate:</font>'
        #                       + '<font color="{hrColor}"> {hr:.0f} [bpm]</font>'
        #                       + '<br><br><font color="black">&nbsp; Respiratory Rate:</font>'
        #                       + '<font color="{respColor}"> {resp:.0f} [bpm]</font>').format(**result))
    
    
    def updateMessageBox(self, name, flag : bool):
        try: 
            if self.messagesToUser_On[name] == flag:
                return
            
            self.messagesToUser_On[name] = flag
            
            to_print = '<b>Message Box:</b>'
            for name, message in self.messagesToUser.items(): 
                if self.messagesToUser_On[name] is True:
                    to_print += '<p>'
                    to_print += message
                    to_print += '</p>'
                    
            self.MessageBoxLabel.setText(to_print)
            
        except AttributeError as e:
            print(e)
        
        except BaseException as e:
            print(e)
        
        except:
            print('unknown error when printing messages')    
    
    
    def reset_plot(self):
        del self.newData
        self.newData = None
        
        axes_to_reset = (self.hrAx, 
                         self.respAx, 
                         self.ppgAx, 
                         self.WelchAx, 
                         self.lombAx)
        
        for ax in axes_to_reset:
            lines = ax.get_lines()
            
            for line in lines:
                line.set_data([], [])
    
    
    def closeEvent(self, event):
        # print(event)
        self.VideoSource.quit()
        self.VideoSource.wait()
        event.accept()
    

    def initUI(self):
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)
        # self.resize(640, 480)
        
        # create a label for live video
        self.label = QLabel(self)
        self.grid_layout.addWidget(self.label, 0, 0, 2, 2) # row, col, hight, width
        self.label.setAlignment(Qt.AlignCenter)
        self.label.resize(640, 480)
        
        # create message box
        self.MessageBoxLabel = QLabel()
        self.grid_layout.addWidget(self.MessageBoxLabel, 0, 3, 2, 1) # row, col, hight, width
        self.MessageBoxLabel.setAlignment(Qt.AlignLeft)
        self.MessageBoxLabel.setStyleSheet("""QLabel { 
                                   background-color : white;
                                   color : black;
                                   font-size : 14pt; 
                                   }""")
        self.updateMessageBox('distance', False)
        
        # Label for hr and rr data
        self.vitals_widget = QWidget()
        self.vitals_grid = QGridLayout()
        self.vitals_widget.setLayout(self.vitals_grid)
        self.grid_layout.addWidget(self.vitals_widget, 0, 2, 2, 1)
        
        
        self.HrLabel = QLabel()
        self.vitals_grid.addWidget(self.HrLabel, 0, 0, 1, 1) # row, col, hight, width
        self.HrLabel.setStyleSheet("""QLabel { 
                                   background-color : white;
                                   color : black;
                                   font-size : 26pt; 
                                   }""")
        self.HrLabel.setAlignment(Qt.AlignCenter)
        
        self.RespLabel = QLabel()
        self.vitals_grid.addWidget(self.RespLabel, 1, 0, 1, 1) # row, col, hight, width
        self.RespLabel.setStyleSheet("""QLabel { 
                                    background-color : white;
                                    color : black;
                                    font-size : 26pt; 
                                    }""")
        self.RespLabel.setAlignment(Qt.AlignCenter)

        self.HrBPMLabel = QLabel()
        self.vitals_grid.addWidget(self.HrBPMLabel, 0, 1, 1, 1) # row, col, hight, width
        self.HrBPMLabel.setPixmap(QPixmap('images/hr.png').scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio))
        self.HrBPMLabel.setAlignment(Qt.AlignCenter)
        self.HrBPMLabel.setStyleSheet('QLabel { background-color : white;}')

        self.RespBPMLabel = QLabel()
        self.vitals_grid.addWidget(self.RespBPMLabel, 1, 1, 1, 1) # row, col, hight, width
        self.RespBPMLabel.setPixmap(QPixmap('images/rr.png').scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio))
        self.RespBPMLabel.setAlignment(Qt.AlignCenter)
        self.RespBPMLabel.setStyleSheet('QLabel { background-color : white;}')
        
        self.updateVitalsDisplay()
        
        # organize buttons in grid
        self.buttons_widget = QWidget()
        self.buttons_grid = QGridLayout()
        self.buttons_widget.setLayout(self.buttons_grid)
        self.grid_layout.addWidget(self.buttons_widget, 2, 0, 3, 2)
        
        # organize ProgressBars in grid
        self.progressbars_widget = QWidget()
        self.progressbars_grid = QGridLayout()
        self.progressbars_widget.setLayout(self.progressbars_grid)
        self.buttons_grid.addWidget(self.progressbars_widget, 0, 0, 1, 4)
        
        # add spinbox for welch history
        self.welchSpinBox = QLabeledSpinBox(label='welch no.\nof windows')
        self.welchSpinBox.setGeometry(0, 0, 40, 20)
        self.buttons_grid.addWidget(self.welchSpinBox, 1, 1, 1, 1, alignment=Qt.AlignBottom)
        
        # add spinbox for resp history
        self.respSpinBox = QLabeledSpinBox(label='lomb no.\nof windows', initValue=6)
        self.respSpinBox.setGeometry(0, 0, 40, 20)
        self.buttons_grid.addWidget(self.respSpinBox, 1, 2, 1, 1, alignment=Qt.AlignBottom)
        
        # add spinbox for welch window size
        self.welchWinSizeSpinBox = QLabeledSpinBox(label='welch\nwindow length', initValue=20, range=(1,30))
        self.welchWinSizeSpinBox.setGeometry(0, 0, 40, 20)
        self.buttons_grid.addWidget(self.welchWinSizeSpinBox, 1, 3, 1, 1, alignment=Qt.AlignBottom)
        
        # add reset button
        self.resetButton = QPushButton('reset')
        self.resetButton.setFixedSize(95,95)
        self.buttons_grid.addWidget(self.resetButton, 1, 0, 1, 1, alignment=Qt.AlignBottom)
        
        # progress bar
        self.snrLevelBar = QLabeledProgressBar(objectName='snr', textVisible=True, label='snr', range=(-5,5), colormap={'green': (0,10), 'red': (-10,0)})
        self.progressbars_grid.addWidget(self.snrLevelBar, 0, 0, 1, 1)
        
        # progress bar
        self.brightnessLevel = QLabeledProgressBar(objectName='brightness', textVisible=True, label='light', range=(0,255), format='{:0.0f}', colormap={'green': (150,256), 'red': (0,150)}, printMessage=self.updateMessageBox)
        self.progressbars_grid.addWidget(self.brightnessLevel, 0, 1, 1, 1)
        
        # progress bar
        self.distanceLevel = QLabeledProgressBar(objectName='distance', textVisible=True, label='prox.', range=(0,1), format='{:.1f}', colormap={'green': (0.4, 2), 'red': (0, 0.4)}, printMessage=self.updateMessageBox)
        self.progressbars_grid.addWidget(self.distanceLevel, 0, 2, 1, 1)
        
        # progress bar
        self.movementLevel = QLabeledProgressBar(objectName='movement', textVisible=True, label='movement', range=(0,3*self.Fs), format='{:0.0f}', colormap={'green': (0, 1*self.Fs), 'red': (1*self.Fs, 6*self.Fs)}, printMessage=self.updateMessageBox)
        self.progressbars_grid.addWidget(self.movementLevel, 0, 3, 1, 1)
        
        
        # add figure for respiratory rate
        self.RespFig = Figure() # figsize=(1, 1.6)) #(4,9)
        self.RespCanvas = FigureCanvas(self.RespFig)
        self.grid_layout.addWidget(self.RespCanvas, 2, 2, 3, 2)
        
        gs = self.RespFig.add_gridspec(3,2)
        self.hrAx = self.RespFig.add_subplot(gs[0, :])
        self.respAx = self.hrAx.twinx()
        self.ppgAx = self.RespFig.add_subplot(gs[1, :])
        self.WelchAx = self.RespFig.add_subplot(gs[2, 0])
        self.lombAx = self.RespFig.add_subplot(gs[2, 1])
        

        COLOR_LOMB = 'magenta'
        COLOR_WELCH = 'blue'
        self.WelchAx.plot([], [], color=COLOR_WELCH)
        self.WelchAx.set_xlabel('beats per minute')
        self.WelchAx.set_title('heart rate: welch', color=COLOR_WELCH)
        self.WelchAx.set_xlim([0, 180])
        self.WelchAx.set_ylim([0, 1.1])
        
        self.lombAx.plot([], [], color=COLOR_LOMB)
        self.lombAx.set_xlim([0, 40])
        self.lombAx.set_ylim([0, 1.1])
        self.lombAx.set_xlabel('breaths per minute')
        self.lombAx.set_title('resp. rate: lomb', color=COLOR_LOMB)
        
        
        # self.ppgAx, self.rriAx, self.lombAx = self.RespFig.subplots(nrows=3, ncols=2)
        self.ppgAx.plot([],[])
        self.ppgAx.plot([], [], "x")
        self.ppgAx.set_xlim([0, self.n_seconds])
        self.ppgAx.set_xlabel('time')
        self.ppgAx.set_title('ppg signal')

        self.hrAx.step([], [], color=COLOR_WELCH)
        self.hrAx.set_xlim([0, 2*self.n_seconds])
        self.hrAx.set_ylim([45, 100])
        self.hrAx.set_xlabel('time')
        self.hrAx.set_ylabel('hr [bpm]', color=COLOR_WELCH)
        self.hrAx.set_title('heart-rate ({}) and respiratory-rate ({})'.format(COLOR_WELCH, COLOR_LOMB))
        
        self.respAx.step([], [], color=COLOR_LOMB)
        self.respAx.set_ylim([5, 25])
        self.respAx.set_ylabel('rr [bpm]', color=COLOR_LOMB)
        
        
        # self.WelchFig.tight_layout()
        self.RespFig.tight_layout()
        
        self.App = App(Fs=self.Fs)
        self.welchSpinBox.connect(self.App.set_welch_nwindows)
        self.respSpinBox.connect(self.App.set_lomb_nwindows)
        self.welchWinSizeSpinBox.connect(self.App.set_welch_nperseg)
        self.resetButton.clicked.connect(self.App.reset)
        self.resetButton.clicked.connect(self.reset_plot)
        
        # self.Welchani = FuncAnimation(self.RespFig, self.WelchUpdate, blit=True, interval=100) 
        self.RespAni = FuncAnimation(self.RespFig, self.RespUpdate, blit=True, interval=100)
        
        self.VideoSource = VideoThread(self.App, Fs=self.Fs)
        self.VideoSource.changePixmap.connect(self.setImage)
        self.VideoSource.changeSnr.connect(self.snrLevelBar.setValue)
        self.VideoSource.changeLight.connect(self.brightnessLevel.setValue)
        self.VideoSource.changeDistance.connect(self.distanceLevel.setValue)
        self.VideoSource.changeMovement.connect(self.movementLevel.setValue)
        self.VideoSource.changeHrResp.connect(self.updateVitalsDisplay)
        self.VideoSource.finished.connect(self.close)
        self.VideoSource.start()
    
    
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    sys.exit(app.exec_())