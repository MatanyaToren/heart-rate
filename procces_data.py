# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.io import savemat, loadmat
# %matplotlib tk
# %%

signal_mat = loadmat('output/signals_09_11_2021_12_21_57_hr_54.mat')

for key, value in signal_mat.items():
    signal_mat[key] = np.squeeze(value)

# %%
raw = signal_mat['raw'] #[2000:-1100]
filtered = signal_mat['filtered'] #[2000:-1100]
t = np.linspace(0, raw.shape[0]/30, raw.shape[0])


# raw signal vs. filtered signal
RawFilteredFig, (RawFilteredAx1, RawFilteredAx2) = plt.subplots(2, 1, sharex=True)

RawFilteredAx1.plot(t, raw)
RawFilteredAx1.set_xlabel('time [sec]')
RawFilteredAx1.set_title('raw signal')
RawFilteredAx1.grid(True)
RawFilteredAx1.xaxis.set_tick_params(labelbottom=True)

RawFilteredAx2.plot(t, filtered)
RawFilteredAx2.set_xlabel('time [sec]')
RawFilteredAx2.set_title('filtered signal')
RawFilteredAx2.grid(True)
RawFilteredAx2.xaxis.set_tick_params(labelbottom=True)

plt.tight_layout()
plt.show()
# RawFilteredFig.savefig('output/RawFilteredSignal.png', dpi=500)


# %%
# filtered signal with peaks, rri signal, lomb periodogram
RespFig, (signalAx, rriAx, lombAx) = plt.subplots(3, 1)

signalAx.plot(t, signal_mat['filtered'])
signalAx.plot(t[signal_mat['peak_times']], signal_mat['filtered'][signal_mat['peak_times']], 'x')
signalAx.set_xlabel('time [sec]')
signalAx.set_title('filtered signal')
signalAx.grid(True)
signalAx.xaxis.set_tick_params(labelbottom=True)

rriAx.stem(t[signal_mat['peak_times']], signal_mat['rri']/30, basefmt='C0-', markerfmt='C0.')
rriAx.set_ylim([0.6, 1.3])
rriAx.set_xlabel('time [sec]')
rriAx.set_title('rri signal')
rriAx.grid(True)
rriAx.sharex(signalAx)
rriAx.xaxis.set_tick_params(labelbottom=True)

lombAx.plot(signal_mat['freqs']*60, signal_mat['pgram'])
lombAx.set_xlim([5, 25])
lombAx.set_xlabel('frequency [bpm]')
lombAx.set_title('lomb-scargle periodogram')
lombAx.grid(True)


plt.tight_layout()
plt.show()

# %%
# filtered signal with peaks, welch periodogram

WelchFig, (WelchSignalAx, WelchAx) = plt.subplots(2, 1)

WelchSignalAx.plot(t, signal_mat['filtered'])
WelchSignalAx.plot(t[signal_mat['peak_times']], signal_mat['filtered'][signal_mat['peak_times']], 'x')
WelchSignalAx.set_xlim([t[-1]-20, t[-1]-5])
WelchSignalAx.set_xlabel('time [sec]')
WelchSignalAx.set_title('filtered signal')
WelchSignalAx.grid(True)

WelchAx.plot(signal_mat['f']*60, signal_mat['pxx'])
WelchAx.set_xlim([40, 120])
WelchAx.set_xlabel('frequency [bpm]')
WelchAx.set_title('welch periodogram')
WelchAx.grid(True)

plt.tight_layout()
plt.show()

# %%
# uneven sampling

time = np.linspace(0, 10, 100)
x = np.sin(2*np.pi*time)

ratio = 1/3
samples = np.random.choice(x.shape[0], size=int(x.shape[0]*ratio), replace=False)
samples.sort()

lombFig, (xAx, fftAx, lombAx) = plt.subplots(3, 1)

xAx.plot(time, x, 'o', fillstyle='none',color='#1f77b4', ms=4 ,label='even')
xAx.plot(time[samples], x[samples], 'o', fillStyle='full', color='#1f77b4', ms=4, label='uneven')
xAx.set_xlabel('time [sec]')
xAx.set_title('$sin(2\pi\cdot t)$')
xAx.set_ylim([-1.3, 1.3])
xAx.legend(loc='lower right')
xAx.grid(True)

fftfreq = np.fft.fftfreq(int(x.shape[0]*ratio), d=10/int(x.shape[0]*ratio))
xfft = np.abs(np.fft.fft(x[samples]))
fftAx.plot(fftfreq[fftfreq >= 0], xfft[fftfreq >= 0])
fftAx.set_xlabel('frequency [Hz]')
fftAx.set_title('fft of unevenly sampled data')
fftAx.grid(True)

lomb = signal.lombscargle(time[samples], x[samples], fftfreq[fftfreq > 0.1]*2*np.pi, precenter=True)
lombAx.plot(fftfreq[fftfreq > 0.1], lomb)
lombAx.set_xlabel('frequency [Hz]')
lombAx.set_title('lomb-scargle of unevenly sampled data')
lombAx.sharex(fftAx)
lombAx.xaxis.set_tick_params(labelbottom=True)
lombAx.grid(True)


plt.tight_layout()
plt.show()


# %%
# nperseg + nstep*(nwindows-1) = 20 + 2*(20-1) = 58 sec

filtered_signal = signal_mat['filtered'][-58*30:]

fft_signal = np.abs(np.fft.fft(filtered_signal))
fftfreq_signal = np.fft.fftfreq(filtered_signal.shape[0], 1/30)

WelchFftFig, (fftSignalAx, WelchSignalAx) = plt.subplots(2, 1, sharex=True)

WelchSignalAx.plot(signal_mat['f'], signal_mat['pxx'])
WelchSignalAx.set_xlabel('frequency [Hz]')
WelchSignalAx.set_title('Welch periodogram')
WelchSignalAx.xaxis.set_tick_params(labelbottom=True)
WelchSignalAx.grid(True)

fftSignalAx.plot(fftfreq_signal[fftfreq_signal >= 0], fft_signal[fftfreq_signal >= 0])
fftSignalAx.set_xlabel('frequency [Hz]')
fftSignalAx.set_title('FFT')
fftSignalAx.xaxis.set_tick_params(labelbottom=True)
fftSignalAx.grid(True)
fftSignalAx.set_xlim([0.5, 3])

plt.tight_layout()
plt.show()



# %%
signal_mat1 = loadmat('output/signals_09_11_2021_16_38_28_far.mat')

for key, value in signal_mat1.items():
    signal_mat1[key] = np.squeeze(value)
    

signal_mat2 = loadmat('output/signals_09_11_2021_16_48_06_far.mat')

for key, value in signal_mat2.items():
    signal_mat2[key] = np.squeeze(value)
    

# %%
FarNearFig, (nearAx, farAx) = plt.subplots(2, 1, sharex=True)

nearAx.plot(signal_mat1['f'], signal_mat1['pxx'])
nearAx.set_xlabel('frequency [Hz]')
nearAx.set_title('Welch of signal from near, proximity indicator: 0.7')
nearAx.xaxis.set_tick_params(labelbottom=True)
nearAx.grid(True)

farAx.plot(signal_mat2['f'], signal_mat2['pxx'])
farAx.set_xlabel('frequency [Hz]')
farAx.set_title('Welch of signal from far, proximity indicator: 0.1')
farAx.xaxis.set_tick_params(labelbottom=True)
farAx.grid(True)
farAx.set_xlim([0.5, 3])

plt.tight_layout()
plt.show()

# %%
