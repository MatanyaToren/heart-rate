import numpy as np
import cv2
import scipy.signal as signal
import matplotlib.pyplot as plt

class respiratory():
    """
    class for estimating resipratory rate from rppg signal
    """
    def __init__(self, n_beats, distance=7, display=False):
        self.peak_times = []
        self.rri = []
        self.freqs = []
        self.pgrams = []
        self.time = 0
        self.distance = distance

        # self.freqs = freqs
        self.n_beats = n_beats
        self.fig = None

        if display:
            self.fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            ax1.plot([], [])
            ax1.plot([], [], "x")
            ax1.set_xlabel('sample')

            ax2.plot([], [])
            ax2.set_xlabel('theta [rad]')

            ax3.plot([], [])
            ax3.set_xlabel('sample time')
            ax3.set_ylabel('rri')

            plt.tight_layout()

    def find_peaks(self, ppg):
        """
        find ppg peaks
        """
        peaks, _ = signal.find_peaks(ppg, distance=self.distance)
        self.peak_times.extend((peaks+self.time).tolist()[1:])
        self.rri.extend(np.diff(peaks).tolist())
        self.time += len(ppg)
        return peaks

    def main(self, ppg):
        """
        main calculation of respiratory rate
        """
        peaks = self.find_peaks(ppg)
        if peaks.shape[0] == 0:
            raise RuntimeError
        
        self.freqs, self.pgram = self.esitmate_res_rate()

        if self.fig is not None:
            self.update_plot(ppg, peaks)

        return self.freqs, self.pgram

    def esitmate_res_rate(self):
        """
        use time differences between peaks to estimate resipratory rate
        """
        f, pgram = respiratory.lomb(self.peak_times[-self.n_beats:], self.rri[-self.n_beats:])
        return f, pgram

    
    def update_plot(self, ppg, peaks):
        xdata = iter([np.arange(self.time-len(ppg), self.time), peaks+self.time-len(ppg), self.freqs, self.peak_times])
        ydata = iter([ppg, ppg[peaks], self.pgram, self.rri])
        for ax in self.fig.get_axes():
            for line in ax.get_lines():
                x = next(xdata)
                y = next(ydata)
                # print(len(x), len(y))
                line.set_data(x, y)
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
        self.fig.canvas.draw_idle()


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
    N = 1000
    n = np.arange(N)
    x = np.sin(2*np.pi/10*n + np.sin(n*2*np.pi/40))  + np.random.randn(N)/5

    # freqs = np.linspace(0.01, np.pi, 50)
    res = respiratory(1000, display=True)
    
    freq, ppx = res.main(x)

    print('respiratory rate is {:.2f} Hz'.format(freq[ppx.argmax()]))
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    # ax1.plot(n, x)
    # ax1.plot(peaks, x[peaks], "x")
    # ax1.set_xlabel('sample')
    #
    # ax2.plot(freqs, pgram)
    # ax2.set_xlabel('theta [rad]')
    #
    # ax3.plot(res.peak_times, res.rri)
    # ax3.set_xlabel('sample time')
    # ax3.set_ylabel('rri')
    
    plt.show()
