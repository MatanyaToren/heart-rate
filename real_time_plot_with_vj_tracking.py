import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal
import scipy.fft as fft
from time import time, sleep
from queue import Queue, Empty
from threading import Thread, Event
from welch_update import welch_update
import os

Fs = 15
nperseg = 12 * Fs
noverlap = 10 * Fs
nstep = nperseg - noverlap

min_bpm, max_bpm = (45, 200)
min_idx = int(min_bpm)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ydata = []
t = np.arange(nperseg)/Fs
ln1, = ax1.plot([], [])

freq = fft.rfftfreq(60*Fs, d=1/Fs) * 60

ax2.set_xlabel('bpm')
ln2, = ax2.plot([], [])

ax3.set_xlabel('bpm')
ax3.set_title('fft')
ln3, = ax3.plot([], [])

FrameQueue = Queue()
SignalQueue = Queue()
FFTQueue = Queue()

stopCapture = Event()

class ani():
    def __init__(self):
        self.ani = FuncAnimation(fig, self.update,
                                init_func=self.init, blit=True, interval=1)

    def init(self):
        ax1.set_xlim(0, nperseg//Fs)
        ax1.set_ylim(-5, 15)

        ax2.set_xlim([0, 180])
        ax2.set_ylim([0, 1.1])

        ax3.set_xlim([0, 180])
        ax3.set_ylim([0, 1.1])

        return ln1, ln2, ln3

    
    def update(self, frame):
        
        x = SignalQueue.get()
        if x is None:
            self.ani.event_source.stop()
            plt.close(fig)
            return ln1, ln2

        ydata.extend(x)
        ln1.set_data(t[:len(ydata)], ydata[-nperseg:])
        ax1.set_ylim([np.min(ydata[-nperseg:]), np.max(ydata[-nperseg:])])

        try:
            p = FFTQueue.get_nowait()
            # print(p.shape)
            ln2.set_data(freq, p)
            ax2.set_ylim([p.min(), p.max()])
            # print(ln2.get_xdata())
        except Empty:
            pass

        fftData = np.abs(np.fft.rfft(ydata[-Fs*30:], 60*Fs))
        ln3.set_data(freq, fftData)
        ax3.set_ylim([0, fftData.max()])

        return ln1, ln2, ln3

def producer():
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture("../videos/MATANYA_63bpm_60sec.mp4")
    
    while stopCapture.is_set() is not True:
        ret, frame = cap.read()
        # frameSmall = cv2.resize(frame, (640, 360))
        if ret:
            FrameQueue.put(frame)
        else:
            break

    SignalQueue.put(None)
    cv2.destroyAllWindows()
    cap.release()
    


def processor():
    
    # Viola & Jones face recognition
    cascPath = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(cascPath)
    detectionRate = 120

    # MOSSE tracker
    tracker = cv2.legacy.TrackerMOSSE_create()


    bandPass = signal.firwin(100, np.array([min_bpm, max_bpm])/60, fs=Fs, pass_zero=False)
    z = np.zeros(bandPass.shape[-1]-1)

    welch_obj = welch_update(fs=Fs, nperseg=nperseg, nwindows=20, nfft=Fs*60)

    ok = False
    (x, y, w, h) = (0, 0, 0, 0)
    raw_signal = []
    filtered_signal = np.array([])
    downSize = 0.6
    global numFrames 
    numFrames = 0
    HeartRate = 0.0

    while True:
        frame = FrameQueue.get()
        
        # use only one out of 30//Fs frames (30 fps camera) 
        numFrames += 1
        if numFrames % (30//Fs) != 0:
            continue

        
        if numFrames % detectionRate == 1 or not ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                x = int(x + w * (1-downSize)/2)
                w = int(w * downSize)
                y = int(y + h * (1-downSize)/2)
                h = int(h * downSize)
                
            

            if h == 0 or w == 0:
                ok = False
                continue

            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, (x, y, w, h))
            ok = True
            # print('reinitiate')
            

        else:
            ok, bbox = tracker.update(frame)
            x, y, wnew, hnew = bbox

            if wnew > 0 and hnew > 0:
                x, y, w, h = int(x), int(y), int(wnew), int(hnew)
            else:
                x, y = int(x), int(y)
            # print('update')
        
        # spatial mean of the bounding box of the face
        if w == 0 or h == 0:
            raise RuntimeError('slice is empty')
        raw_signal.append(np.mean(frame[x:x+w, y:y+w, 1][:]))
        

        if 0 < len(raw_signal) and 0 == len(raw_signal) % 10:
            filtered_chunk, z = signal.lfilter(bandPass, 1, raw_signal[-10:], zi=z)
            SignalQueue.put(filtered_chunk)
            filtered_signal = np.concatenate((filtered_signal, filtered_chunk))

        if nperseg < filtered_signal.shape[0] and 0 == filtered_signal.shape[0] % nstep:
            f, pxx = welch_obj.update(filtered_signal[-nperseg:])
            FFTQueue.put(pxx)
            HeartRate = f[np.argmax(pxx)] * 60
            # print(HeartRate)
            # print(f.shape, f[np.argmax(pxx)])
            
        frameRect = cv2.flip(cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2), 1)
        cv2.putText(frameRect, "Heart Rate: {:.1f} bpm".format(HeartRate), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.imshow('frame', frameRect[::3, ::3, :])    
        FrameQueue.task_done()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            SignalQueue.put(None)
            stopCapture.set()
            break
    
    

        

Thread(target=producer, name='webcam thread', args=(), daemon=True).start()
Thread(target=processor, name='webcam view thread', args=(), daemon=True).start()

start = time()
a = ani()


plt.show()
print('Fs: {:.2f}'.format(numFrames/(time()-start)))
