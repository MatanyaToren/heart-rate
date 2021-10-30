import numpy as np
from scipy.signal import welch
from scipy.signal.windows import get_window


class welch_update:
    def __init__(self, nwindows=5, fs=1.0, window='hann', nperseg=256, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, average='mean'):

        self.set_nwindows(nwindows)
        
        self.nwindow = nwindows
        self.fs=fs
        self.window_type = window
        self.window = get_window(window, nperseg)
        self.nperseg = nperseg
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.average = average

        self.windows = []


    def update(self, x):
        if x.shape[0] != self.nperseg:
            # raise ValueError("every segment must have the same length as 'nperseg'")
            self.window = get_window(self.window_type, x.shape[0])

        f, welch_segment = welch(x, fs=self.fs, window=self.window, nperseg=x.shape[0], noverlap=None, nfft=self.nfft,
                                detrend=self.detrend, return_onesided=self.return_onesided, scaling=self.scaling,
                                axis=self.axis, average=self.average)
        self.windows.append(welch_segment)

        while self.nwindow < len(self.windows):
            del self.windows[0]

        # print('welch no. windows: ', len(self.windows))
        return f, np.mean(self.windows, axis=0)

    def set_nwindows(self, nwindows):
        if 0 != nwindows % 1 and 0 >= nwindows:
            raise ValueError('nwindows (number of windows to average) should be a natuarl number')
        
        self.nwindow = nwindows          
