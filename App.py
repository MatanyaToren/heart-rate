import numpy as np
from scipy import fft
import scipy.signal as signal
from queue import Queue, Empty
from threading import Thread, Event
from tracking import *
from respiratory_rate import *
from welch_update import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage

import os
import sys


class App(QThread):
    """
    class for running heart rate and respiratory rate app
    """
    changePixmap = pyqtSignal(QImage)
    
    def __init__(self, Fs=15, detectionRate=120, n_beats=60, distance=2/3, min_bpm=45, max_bpm=200,
                 nperseg=12, noverlap=10, resp_nstep=10):
        
        super().__init__()
        self.Fs = Fs
        self.min_bpm, self.max_bpm = min_bpm, max_bpm
        self.nperseg = nperseg * Fs
        self.noverlap = noverlap * Fs
        self.nstep = (nperseg - noverlap) * Fs
        self.resp_nstep = resp_nstep * Fs  # updating rate of the respiratory rate

        self.offset_x = 0.1
        self.offset_y = 0.6
        self.size_x = 0.2
        self.size_y = 0.2

        self.raw_signal = []
        self.filtered_signal = []
        
        self.tracker = FaceTracker(detectionRate=detectionRate)
        self.welch_obj = welch_update(fs=self.Fs, nperseg=self.nperseg)
        self.resp = respiratory(fs=self.Fs, n_beats=n_beats, distance=int(distance*Fs))
        self.bandPass = signal.firwin(100, np.array([min_bpm, max_bpm])/60, fs=Fs, pass_zero=False)
        self.z = np.zeros(self.bandPass.shape[-1]-1)
        
        self.FrameQueue = Queue()
        self.welch_data_available = Event()
        self.resp_data_available = Event()
        self.stop = Event()
        self.CameraThread = Thread(target=self.get_frames, name='CameraThread', args=(), daemon=False)
        
        
    def run(self):
        """
        run the app
        """
        self.CameraThread.start()
        self.n = 0
        self.NumFrames = 0
        self.HeartRate = 0
        self.RespRate = 0
        self.bbox = (0,0,0,0)
        self.roi = (0,0,0,0)
        
        while self.stop.is_set() is not True:
            frame = self.FrameQueue.get()
            
            self.NumFrames += 1
            if self.NumFrames % (30//self.Fs) != 0:
                continue
        
            try:
                self.bbox = self.tracker.update(frame)

            except (TrackingError, OutOfFrameError, DetectionError) as err:
                # print(err.message)
                self.to_display(frame)
                continue
            
            # self.get_signal()
            # self.n += 1
            
            # if 0 < self.n and 0 == self.n % self.nstep:
            #     filtered_chunk, self.z = signal.lfilter(self.bandPass, 1, self.raw_signal[-10:], zi=self.z)
            #     self.filtered_signal.extend(filtered_chunk.tolist())
                
            # if self.nperseg < self.n and 0 == self.n % self.nstep:
            #     self.f, self.pxx = self.welch_obj.update(np.array(self.filtered_signal[-self.nperseg:]))
            #     self.HeartRate = self.f[np.argmax(self.pxx)] * 60
            #     self.welch_data_available.set()
        

            # if self.resp_nstep < self.n and 1 == self.n % self.resp_nstep:
            #     # calculate the respiratory rate
            #     self.freqs, self.pgram = self.resp.main(self.filtered_signal[-self.resp_nstep:])
            #     self.RespRate = self.freqs[self.pgram.argmax()] * self.Fs * 60
            #     self.resp_data_available.set()
            
            
            self.to_display(frame)

        self.CameraThread.join()    
        
    def get_frames(self):
        """
        a seperate thread to get frames from camera or file
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap = cv2.VideoCapture('videos/56bpm_17_08.mp4')
    
        # Read first frames
        # for i in range(5):
        #     ret, frame = cap.read()
        #     if not ret:
        #         print('Cannot read video file')
        #         self.stop.set()
        #         cap.release()
        #         return


        while self.stop.is_set() is not True:
            ret, frame = cap.read()
            if ret:
                self.FrameQueue.put(frame)
            else:
                self.stop.set()
                print('cant capture frame')
                cap.release()
                return

        
    
    
    def get_signal(self):
        """
        get signal from roi
        """
        x, y, w, h = self.bbox
        
        x_bb = int(x + self.offset_x * w)
        w_bb = int(w * self.size_x)
        y_bb = int(y + self.offset_y * h)
        h_bb = int(h * self.size_y)
        
        self.roi = (x_bb, y_bb, w_bb, h_bb)
        
        # spatial mean of the bounding box of the face
        self.raw_signal.append(np.mean(self.frame[x_bb:x_bb+w_bb, y_bb:y_bb+w_bb, 1][:]))

    
    def to_display(self, frame):
        """
        add bbox and roi to frame and send to display
        """
        x, y, w, h = self.bbox
        x_bb, y_bb, w_bb, h_bb = self.roi
        
        frameRect = cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2, 1)
        frameRect = cv2.flip(cv2.rectangle(frameRect, (x_bb, y_bb), (x_bb+w_bb, y_bb+h_bb), (0, 255, 0), 2), 1)
        cv2.putText(frameRect, "Heart Rate: {:.1f} bpm".format(self.HeartRate), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.putText(frameRect, "Breathing Rate: {:.1f} bpm".format(self.RespRate), (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        rgbImage = cv2.cvtColor(frameRect, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        # print('frame sent to display')
        self.changePixmap.emit(p)
    
    def quit(self):
        self.stop.set()