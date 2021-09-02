import numpy as np
import cv2
import scipy.signal as signal

class respiratory():
    """
    class for estimating resipratory rate from rppg signal
    """
    def __init__(self, n_beats):
        self.peak_times = []
        self.rri = []
        self.time = 0

        # self.freqs = freqs
        self.n_beats = n_beats

    
    def find_peaks(self, ppg):
        """
        find ppg peaks
        """
        peaks, _ = signal.find_peaks(ppg)
        self.peak_times.extend((peaks+self.time).tolist()[1:])
        self.rri.extend(np.diff(peaks).tolist())
        self.time += len(ppg)
        return peaks


    def esitmate_res_rate(self):
        """
        use time differences between peaks to estimate resipratory rate
        """
        f, pgram = respiratory.lomb(self.peak_times[-self.n_beats:], self.rri[-self.n_beats:])
        return f, pgram

    def lomb(t, y):
        """Compute the Lomb-Scargle periodogram

        Inputs
            t : sample times
            y : measurement values

        Outputs
            f   : frequency vector
            pxx : power spectral density (PSD) estimate

        Inputs are assumed to be vectors of type numpy.ndarray.
        """
        n = len(t)
        ofac = 4  # Oversampling factor
        hifac = 1
        T = t[-1] - t[0]
        Ts = T / (n - 1)
        nout = np.round(0.5 * ofac * hifac * n) 
        f = np.arange(1, nout+1) / (n * Ts * ofac)
        f_ang = f * 2 * np.pi
        pxx = signal.lombscargle(t, y, f_ang, precenter=True)
        pxx = pxx * 2 / n
        return f, pxx


if __name__ == '__main__':
    n = np.arange(1000)
    x = np.sin(np.pi/10*n + np.pi*np.sin(n*np.pi/80)) # + np.random.randn(1000)/5

    # freqs = np.linspace(0.01, np.pi, 50)
    res = respiratory(1000)
    peaks = res.find_peaks(x)

    freqs, pgram = res.esitmate_res_rate()

    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(n, x)
    ax1.plot(peaks, x[peaks], "x")
    ax1.set_xlabel('sample')

    ax2.plot(freqs, pgram)
    ax2.set_xlabel('theta [rad]')

    ax3.plot(res.peak_times, res.rri)
    ax3.set_xlabel('sample time')
    ax3.set_ylabel('rri')

    plt.show()