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
        self.noverlap = 18 * Fs # this parameter is not used
        self.nstep = self.nperseg - self.noverlap
        self.resp_nstep = 10 * Fs  # updating rate of the respiratory rate
        self.filter_step = int(np.gcd(self.nstep, self.resp_nstep) / 6)

        self.HeartRate = [0]
        self.HeartRateValid = [False]
        self.HeartRateTime = [0]
        self.RespRate = [0]
        self.RespRateValid = [False]
        self.RespRateTime = [0]
        self.n = 0
        self.raw_signal = []
        self.filtered_signal = []
        self.brightness = ([0], [0], [0])
        self.distance_ratio = ([0], [0], [0])
        self.snr = [0]
        self.diff_center_face = [0]
        self.movement_indicator = [0]
        self.middle_x = None
        self.middle_y = None
        
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
            self.get_signal(frame)
        
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
           
        self.get_movements()
        self.get_brightness(frame)
        self.get_distance_indicator(frame)
        self.n += 1
        
        if self.bandPass.shape[0] <= self.n and 0 == self.n % self.filter_step:
            # filtered_chunk, self.z = signal.lfilter(self.bandPass, 1, self.raw_signal[-10:], zi=self.z)
            # self.filtered_signal = np.concatenate((self.filtered_signal, filtered_chunk))
            self.filtered_signal = signal.filtfilt(self.bandPass, 1, self.raw_signal, padlen=min(len(self.raw_signal)-2, 3*self.bandPass.shape[0]))
            self.SignalQueue.put(self.filtered_signal)
            

        if max(self.nperseg, self.bandPass.shape[0]) <= self.n and 0 == self.n % self.nstep:
            f, pxx = self.welch_obj.update(self.filtered_signal[-self.nperseg:])
            hr_valid_range = np.logical_and(f >= self.min_bpm/60, f <= self.max_bpm/60)
            tmp_hr = f[hr_valid_range][np.argmax(pxx[hr_valid_range])] * 60
            HeartRate, HeartRateValid = self.heart_rate_otlier_removal.update(tmp_hr)
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
                
                if 40*self.Fs <= self.n:
                    pgram = pgram * np.logical_and(freqs >= 5/self.Fs/60, freqs <= 25/self.Fs/60)
                    resp_peaks, _ = signal.find_peaks(pgram)
                    sorted_args = np.argsort(pgram[resp_peaks])
                    max_peak = resp_peaks[sorted_args[-1]]
                    second_peak = resp_peaks[sorted_args[-2]]
                                            
                    RespRate, RespRateValid = self.resp_rate_otlier_removal.update(freqs[max_peak] * self.Fs * 60)
                    self.RespRate.append(RespRate)
                    self.RespRateValid.append(RespRateValid and (pgram[max_peak] > 1.5*pgram[second_peak]))
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
        rois = []
        for (offset_x, offset_y, size_x, size_y) in self.offsets:
            x_roi = int(x + offset_x * w)
            w_roi = int(w * size_x)
            y_roi = int(y + offset_y * h)
            h_roi = int(h * size_y)
            
            if (x_roi < 0 or x_roi+w_roi+1 >= frame.shape[1] 
                or y_roi < 0 or y_roi+h_roi+1 >= frame.shape[0]):
                
                raise OutOfFrameError
            
            rois.append((x_roi, y_roi, w_roi, h_roi))
            
            try:
                # spatial mean of the bounding box of the face
                newSample += np.mean(frame[y_roi:y_roi+h_roi+1, x_roi:x_roi+w_roi+1, 1][:]) #\
                            # - np.mean(frame[y_roi:y_roi+h_roi+1, x_roi:x_roi+w_roi+1, 2][:])
                            
            except Exception as err:
                print(err)
                
            self.rois = rois
                        
        self.raw_signal.append(newSample)



    def get_brightness(self, frame):
        """
        This function computes the brightness in the rois, for quality assurance purposes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        try:
            frame_area = frame.shape[0] * frame.shape[1]
            for (_, _, w_roi, h_roi), ratio in zip(self.rois, self.distance_ratio):
                roi_area = w_roi * h_roi
                ratio.append(roi_area / (frame_area + 1e-7) * (100))
                
        except RuntimeWarning as e:
            print(e)
            raise e
            
        except:
            print('error in distance computation')
            raise Exception

        return self.distance_ratio
    
    
    def get_snr(self, pxx, f, peak):
        """
        This function computes the snr of the signal received.txt
        The snr in this context is defines as the 10*log_10 of the ratio of the Energy of pxx in 
        the 0.2 Hz around the highest peak and the total energy of the signal
        """
        try:
            TotalEnergy = pxx.sum()
            SignalRange = np.logical_and(f > (peak - 0.2), f < (peak + 0.2))
            SignalEnergy = pxx[SignalRange].sum()
            snr = 10*np.log10(SignalEnergy / (TotalEnergy - SignalEnergy + 1e-7))
            self.snr.append(snr)
        
        except RuntimeWarning as e:
            print(e)
            raise e
            
        except:
            print('error in snr computation')
            raise Exception
            
        return snr
            
            
    def get_movements(self):
        """
        get signal from roi
        """
        x, y, w, h = self.bbox
        
        middle_x = x + w / 2
        middle_y = y + h / 2
        
        if self.middle_x is not None and self.middle_y is not None:
            self.diff_center_face.append(np.sqrt((self.middle_x - middle_x)**2 
                                                 + (self.middle_y - middle_y)**2))
            # describes pixel displacement of face in a second
            self.movement_indicator.append(np.mean(self.diff_center_face[-3*self.Fs:]) * self.Fs)
            
        else:
            self.middle_x = middle_x
            self.middle_y = middle_y
            return None
        
        self.middle_x = middle_x
        self.middle_y = middle_y
        
        return  self.movement_indicator[-1]
    
    def set_welch_nwindows(self, nwindows):
        self.welch_obj.set_nwindows(nwindows)
        # print('nwindows: ', nwindows)
    
    def set_lomb_nwindows(self, nwindows):
        self.resp.set_nwindows(nwindows)
        # print('nwindows: ', nwindows)
        
    def set_welch_nperseg(self, nperseg : int = 20):
        if nperseg % 1 != 0 or nperseg < 1:
            raise ValueError('nperseg should be an integer larger than 1')
        
        self.nperseg = nperseg * self.Fs
        self.noverlap = self.nperseg - self.nstep
        
        
    def quit(self):
        pass
    
    def reset(self):
        """"
        This function erases the history of the system, 
        but does not changes settings defined by user
        """
        
        # reset queues to display results
        while not self.SignalQueue.empty():
            self.SignalQueue.get_nowait()
            
        while not self.WelchQueue.empty():
            self.SignalQueue.get_nowait()
            
        while not self.SignalQueue.empty():
            self.RespQueue.get_nowait()
        
        
        self.HeartRate = [0]
        self.HeartRateValid = [False]
        self.HeartRateTime = [0]
        self.RespRate = [0]
        self.RespRateValid = [False]
        self.RespRateTime = [0]
        self.n = 0
        self.raw_signal = []
        self.filtered_signal = []
        self.brightness = ([0], [0], [0])
        self.distance_ratio = ([0], [0], [0])
        self.snr = [0]
        
        # reset objects
        # self.roi_finder = roi(types=['all'])
        self.resp = respiratory(n_beats=40, distance=int(self.Fs/2), nwindows=self.resp.nwindows)
        self.welch_obj = welch_update(fs=self.Fs, nperseg=self.nperseg, nwindows=self.welch_obj.nwindow, nfft=self.Fs*60)
        self.heart_rate_otlier_removal = VarianceFilter()
        self.resp_rate_otlier_removal = VarianceFilter()