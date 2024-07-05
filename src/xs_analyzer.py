import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal import chirp
from scipy.signal.windows import flattop, hamming, kaiser, blackman, hann
from constants import Labels, Windows
from quantiphy import Quantity


class Signal():
    def __init__(self, sample_rate=100e6, span=100e6, center_frequency=2e6):
        self.center_frequency = self.check_frequency(center_frequency)
        self.sample_rate = sample_rate
        self.span = span
        # create figure ax 1 and 2 and 3
        self.fig, [self.ax1, self.ax2, self.ax3] = plt.subplots(nrows=3, ncols=1)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(8)

    def calc_sample_time_interval(self):
        t_step = 1 / self.sample_rate
        return t_step

    def check_frequency(self, frequency):
        if -50e6 <= frequency <= 50e6:
            return frequency
        else:
            print("Center frequency = {0} Hz does not match requirement to be within -50Mhz to 50Mhz.".format(frequency))
            exit(1)

    def calculate_number_of_samples(self, freq):
        return self.sample_rate / freq

    def employ_window_fft(self, window_function, points):
        win = None
        if Windows.Bartlett in window_function:
            win = np.bartlett(points)
        elif Windows.Kaiser in window_function:
            win = np.kaiser(points, 55)
        elif Windows.Hamming in window_function:
            win = np.hamming(points)
        elif Windows.Hanning in window_function:
            win = np.hanning(points)
        elif Windows.Blackman in window_function:
            win = np.blackman(points)
        return win

    def waveform(self, span=100e6, fft_size=32, window_type=Windows.Kaiser, sweep=False):
        start_freq = self.center_frequency - span/2
        mid_f1 = self.center_frequency - span/4
        mid_f2 = self.center_frequency + span/4
        end_freq = self.center_frequency + span/2
        sample_time_interval = 1 / self.sample_rate
        number_of_samples = int(self.sample_rate / self.center_frequency)
        time_steps = np.linspace(0, (number_of_samples-1)*sample_time_interval, number_of_samples)
        freq_inteval = self.sample_rate / number_of_samples
        freq_steps = np.linspace(0, (number_of_samples - 1) * freq_inteval, number_of_samples)

        if sweep:
            y = chirp(time_steps, f0=mid_f1, f1=end_freq, t1=(number_of_samples-1)*sample_time_interval, method='linear')
            self.plot_signal(time_steps, y, self.ax1, title="Signal Frequency Sweep",
                             xlabel="Time (s)", ylabel="Amplitude")
        else:
            #  sine waves combined
            y = (1 * np.sin(2 * np.pi * start_freq * time_steps) +
                 8 * np.sin(2 * np.pi  * mid_f1 * time_steps) +
                 4 * np.sin(2 * np.pi * self.center_frequency * time_steps) +
                 16 * np.sin(2 * np.pi * mid_f2 * time_steps) +
                 24 * np.sin(2 * np.pi * end_freq * time_steps))
            self.plot_signal(time_steps, y, self.ax1, title="Signal Sine Wave", xlabel="Time (s)", ylabel="Amplitude")

        # fft windows

        x_signal = np.fft.fft(y)
        x_mag = np.abs(x_signal)
        x_mag = np.fft.fftshift(x_mag)
        f_plot = freq_steps[0:int(number_of_samples/2+1)]
        x_mag_plot = 2 * x_mag[0:int(number_of_samples/2+1)]
        x_mag_plot[0] = x_mag_plot[0] / 2
        q_span = Quantity(self.span, 'Hz')
        q_rbw = Quantity(self.sample_rate/number_of_samples, 'Hz')
        self.plot_signal(f_plot, x_mag_plot, self.ax2,
                         title="Spectrum Analyzer. Span = {0} | RBW = {1}".format(q_span, q_rbw),
                         xlabel="Frequency (Hz)", ylabel="Amplitude")

        # fft - apply kaiser window

        # Generate a signal
        N = fft_size  # Number of samples
        fs = self.sample_rate # Sampling frequency
        t = np.arange(N) / fs

        #  sine waves combined with fft size applied
        y_fft_sized = (1 * np.sin(2 * np.pi * start_freq * t) +
                       8 * np.sin(2 * np.pi * mid_f1 * t) +
                       4 * np.sin(2 * np.pi * self.center_frequency * t) +
                       16 * np.sin(2 * np.pi * 10 * mid_f2 * t) +
                       24 * np.sin(2 * np.pi * end_freq * t))

        # Apply the Kaiser window
        window = self.employ_window_fft(window_type, fft_size)
        windowed_signal = y_fft_sized * window

        # Perform Fourier transform
        spectrum = np.fft.fft(windowed_signal)
        spectrum = np.fft.fftshift(spectrum)
        spectrum = np.abs(spectrum)
        response = 20 * np.log10(spectrum)
        freq = np.fft.fftfreq(N, 1 / fs)
        freq = np.fft.fftshift(freq)

        # Plot the spectrum
        q_rbw = Quantity(self.sample_rate/N, 'Hz')
        self.plot_signal(freq, response, self.ax3,
                             title="{0} Window FFT with FFT Size {1} points. RBW ~ {2}".
                             format(window_type[:1].upper()+window_type[1:], fft_size, q_rbw),
                             xlabel="Frequency (Hz)", ylabel="Magnitude")

    def plot_signal(self, inteval, signal, ax, title, xlabel, ylabel):
        # create plot with labels and title
        ax.plot(inteval, signal, ".-", lw=0.4, c='blue')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        ax.grid()
        if 'Window' not in title:
            ax.set_xlim(0, (inteval[-1]))
        else:
            ax.set_xlim(0, self.sample_rate / 16)
        plt.tight_layout()


if __name__ == '__main__':
    sig = Signal()
    (sig.waveform(span=100e6, fft_size=4096, window_type=Windows.Kaiser, sweep=False))
    plt.show()






