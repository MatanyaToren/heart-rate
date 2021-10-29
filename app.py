import numpy as np
import scipy.signal as signal
from threading import Thread
from queue import Queue, Empty
from tracking import *
from get_roi import *
from welch_update import *
from respiratory_rate import *
from filter_variance import *


class SampleError(RuntimeError):
    def __init__(self):
        self.message = "Couldn't get sample"


class App():
    roi_finder = roi(types=['all'])
    
    def __init__(self, Fs=30, min_bpm=45, max_bpm=200):
        
        # queues to display results
        self.SignalQueue = Queue()
        self.WelchQueue = Queue()
        self.RespQueue = Queue()
    
    
        self.min_bpm, self.max_bpm = min_bpm, max_bpm
        self.offsets = [(0.1, 0.6, 0.2, 0.2), 
                        (0.7, 0.6, 0.2, 0.2), 
                        (0.3, 0.0, 0.2, 0.2)]
        self.Fs = Fs
        self.nperseg = 20 * Fs
        self.noverlap = 18 * Fs
        self.nstep = self.nperseg - self.noverlap
        self.resp_nstep = 10 * Fs  # updating rate of the respiratory rate


        self.HeartRate = [0]
        self.HeartRateValid = [False]
        self.HeartRateTime = [0]
        self.RespRate = [0]
        self.RespRateValid = [False]
        self.RespRateTime = [0]
        self.n = 0
        self.raw_signal = []
        self.filtered_signal = []
        self.brightness = ([], [], [])
        self.distance_ratio = ([], [], [])
        self.snr = [0]
        
        self.bandPass = signal.firwin(200, np.array([min_bpm, max_bpm])/60, fs=Fs, pass_zero=False)
        self.z = 4*np.ones(self.bandPass.shape[-1]-1)
        
        self.tracker = FaceTracker()
        # self.roi_finder = roi(types=['all'])
        self.resp = respiratory(n_beats=40, distance=int(Fs/2), nwindows=6)
        self.welch_obj = welch_update(fs=Fs, nperseg=self.nperseg, nwindows=20, nfft=Fs*60)
        self.heart_rate_otlier_removal = VarianceFilter()
        self.resp_rate_otlier_removal = VarianceFilter()

           
        
    def new_sample(self, frame):
        try:
            self.bbox = self.tracker.update(frame)
        
        except (TrackingError, OutOfFrameError, DetectionError) as err:
            # print(err.message)
            raise SampleError
        
        except JumpingError:
            print("either you're moving too fast or another face entered the view of the camera,")
            print("the app supports only one person at a time")
            raise SampleError
        except Exception as err:
            print('unknown error in tracking')
            # print(err)
           
        self.get_signal(frame)
        self.get_brightness(frame)
        self.get_distance_indicator(frame)
        self.n += 1
        
        if self.bandPass.shape[0] <= self.n and 0 == self.n % 10:
            # filtered_chunk, self.z = signal.lfilter(self.bandPass, 1, self.raw_signal[-10:], zi=self.z)
            # self.filtered_signal = np.concatenate((self.filtered_signal, filtered_chunk))
            self.filtered_signal = signal.filtfilt(self.bandPass, 1, self.raw_signal, padlen=min(len(self.raw_signal)-2, 3*self.bandPass.shape[0]))
            self.SignalQueue.put(self.filtered_signal)
            

        if max(self.nperseg, self.bandPass.shape[0]) <= self.n and 0 == self.n % self.nstep:
            f, pxx = self.welch_obj.update(self.filtered_signal[-self.nperseg:])
            HeartRate, HeartRateValid = self.heart_rate_otlier_removal.update(f[np.argmax(pxx)] * 60)
            self.get_snr(pxx, f, self.HeartRate[-1]/60)
            self.HeartRate.append(HeartRate)
            self.HeartRateValid.append(HeartRateValid)
            self.HeartRateTime.append(self.n/self.Fs)
            self.WelchQueue.put({'f': f, 'pxx': pxx, 
                                 'HeartRate': np.array(self.HeartRate), 
                                 'HeartRateValid': np.array(self.HeartRateValid), 
                                 'HeartRateTime': np.array(self.HeartRateTime), 
                                 'Lower': self.heart_rate_otlier_removal.lower, 
                                 'Higher': self.heart_rate_otlier_removal.higher})


        if max(self.resp_nstep, self.bandPass.shape[0]) <= self.n and 0 == self.n % self.resp_nstep:
            if self.resp.time == 0:
                self.resp.set_time(max(0, self.n-self.resp_nstep))
            # calculate the respiratory rate
            try:
                freqs, pgram = self.resp.main(self.filtered_signal[-self.resp_nstep:])
                pgram = pgram * scipy.stats.norm(14/60, 4/60).pdf(freqs*self.Fs)
                RespRate, RespRateValid = self.resp_rate_otlier_removal.update(freqs[pgram.argmax()] * self.Fs * 60)
                self.RespRate.append(RespRate)
                self.RespRateValid.append(RespRateValid)
                self.RespRateTime.append(self.n/self.Fs)
                
                self.RespQueue.put({'freqs': freqs*self.Fs, 
                                    'pgram': pgram, 
                                    'peak_times': np.array(self.resp.peak_times), 
                                    'rri': np.array(self.resp.rri),
                                    'RespRate': np.array(self.RespRate), 
                                    'RespRateValid': np.array(self.RespRateValid), 
                                    'RespRateTime': np.array(self.RespRateTime), 
                                    'Lower': self.resp_rate_otlier_removal.lower, 
                                    'Higher': self.resp_rate_otlier_removal.higher})
                
            except RuntimeError:
                print('no peaks were found')

             
            
    def get_signal(self, frame):
        """
        get signal from roi
        """
        x, y, w, h = self.bbox
        
        if self.n % 60 == 0:
            self.offsets = self.roi_finder.get_roi(frame, self.bbox)    
        
        newSample = 0
        self.rois = []
        for (offset_x, offset_y, size_x, size_y) in self.offsets:
            x_roi = int(x + offset_x * w)
            w_roi = int(w * size_x)
            y_roi = int(y + offset_y * h)
            h_roi = int(h * size_y)
            
            self.rois.append((x_roi, y_roi, w_roi, h_roi))
            
            try:
                # spatial mean of the bounding box of the face
                newSample += np.mean(frame[y_roi:y_roi+h_roi+1, x_roi:x_roi+w_roi+1, 1][:]) #\
                            # - np.mean(frame[y_roi:y_roi+h_roi+1, x_roi:x_roi+w_roi+1, 2][:])
                            
            except Exception as err:
                print(err)
                
                        
        self.raw_signal.append(newSample)



    def get_brightness(self, frame):
        """
        This function computes the brightness in the rois, for quality assurance purposes
        """
        gray = cv2.cvtColor(frame, cv2. COLOR_BGR2GRAY)
        for (x_roi, y_roi, w_roi, h_roi), brightness in zip(self.rois, self.brightness):
            
            try:
                # spatial mean of the bounding box of the face
                brightness.append(np.mean(gray[y_roi:y_roi+h_roi+1, x_roi:x_roi+w_roi+1][:]))
                
            except RuntimeWarning:
                brightness.append(0)
                
            except Exception as err:
                print(err)

        return self.brightness
                        

    def get_distance_indicator(self, frame):
        """
        This function computes the area ratio between the rois and the frame, for quality assurance purposes
        """
        frame_area = frame.shape[0] * frame.shape[1]
        for (_, _, w_roi, h_roi), ratio in zip(self.rois, self.distance_ratio):
            roi_area = w_roi * h_roi
            ratio.append(roi_area / (frame_area + 1e-7) * (100))

        return self.distance_ratio
    
    
    def get_snr(self, pxx, f, peak):
        """
        This function computes the snr of the signal received.txt
        The snr in this context is defines as the 10*log_10 of the ratio of the Energy of pxx in 
        the 0.2 Hz around the highest peak and the total energy of the signal
        """
        
        TotalEnergy = pxx.sum()
        SignalRange = np.logical_and(f > (peak - 0.2), f < (peak + 0.2))
        SignalEnergy = pxx[SignalRange].sum()
        snr = 10*np.log10(SignalEnergy / (TotalEnergy - SignalEnergy + 1e-7))
        self.snr.append(snr)
        return snr
            
    
    def quit(self):
        pass
    
    