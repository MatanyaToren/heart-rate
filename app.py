import numpy as np
import scipy.signal as signal
from threading import Thread
from queue import Queue, Empty
from tracking import *
from welch_update import *
from respiratory_rate import *


class SampleError(RuntimeError):
    def __init__(self):
        self.message = "Couldn't get sample"


class App():
    def __init__(self, Fs=30, min_bpm=45, max_bpm=200):
        
        # queues to display results
        self.SignalQueue = Queue()
        self.WelchQueue = Queue()
        self.RespQueue = Queue()
    
    
        self.min_bpm, self.max_bpm = min_bpm, max_bpm
        self.offset_x = 0.1
        self.offset_y = 0.6
        self.size_x = 0.2
        self.size_y = 0.2
        self.Fs = Fs
        self.nperseg = 12 * Fs
        self.noverlap = 10 * Fs
        self.nstep = self.nperseg - self.noverlap
        self.resp_nstep = 10 * Fs  # updating rate of the respiratory rate


        self.HeartRate = 0
        self.RespRate = 0
        self.n = 0
        self.raw_signal = []
        self.filtered_signal = []
        
        self.bandPass = signal.firwin(100, np.array([min_bpm, max_bpm])/60, fs=Fs, pass_zero=False)
        self.z = 4*np.ones(self.bandPass.shape[-1]-1)
        
        self.tracker = FaceTracker()
        self.resp = respiratory(n_beats = 60, distance = int(2*Fs/3))
        self.welch_obj = welch_update(fs=Fs, nperseg=self.nperseg, nwindows=20, nfft=Fs*60)

           
        
    def new_sample(self, frame):
        try:
            self.bbox = self.tracker.update(frame)
        
        except (TrackingError, OutOfFrameError, DetectionError) as err:
            raise SampleError
           
        self.get_signal(frame)
        self.n += 1
        
        if self.bandPass.shape[0] <= self.n and 0 == self.n % 10:
            # filtered_chunk, self.z = signal.lfilter(self.bandPass, 1, self.raw_signal[-10:], zi=self.z)
            # self.filtered_signal = np.concatenate((self.filtered_signal, filtered_chunk))
            self.filtered_signal = signal.filtfilt(self.bandPass, 1, self.raw_signal, padlen=min(len(self.raw_signal)-2, 3*self.bandPass.shape[0]))
            self.SignalQueue.put(self.filtered_signal)

        if self.nperseg <= self.n and 0 == self.n % self.nstep:
            f, pxx = self.welch_obj.update(self.filtered_signal[-self.nperseg:])
            self.WelchQueue.put((f, pxx))
            self.HeartRate = f[np.argmax(pxx)] * 60
            # print(HeartRate)
            # print(f.shape, f[np.argmax(pxx)])

        if self.resp_nstep <= self.n and 0 == self.n % self.resp_nstep:
            # calculate the respiratory rate
            freqs, pgram = self.resp.main(self.filtered_signal[-self.resp_nstep:])
            self.RespQueue.put({'freqs': freqs*self.Fs, 'pgram': pgram, 'peak_times': np.array(self.resp.peak_times), 'rri': np.array(self.resp.rri)})
            self.RespRate = freqs[pgram.argmax()] * self.Fs * 60

             
            
    def get_signal(self, frame):
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
        self.raw_signal.append(np.mean(frame[x_bb:x_bb+w_bb, y_bb:y_bb+w_bb, 1][:]))
        
    
    def quit(self):
        pass
    
    