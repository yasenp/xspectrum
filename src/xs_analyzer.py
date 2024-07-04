import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal.windows import flattop, hamming, kaiser, blackman, hann
from constants import Labels, Windows


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

    def sine_wave(self, span=100e6, fft_size=32):
        start_freq = self.center_frequency - span/2
        mid_f1 = self.center_frequency - span/4
        mid_f2 = self.center_frequency + span/4
        end_freq = self.center_frequency + span/2
        sample_time_interval = 1 / self.sample_rate
        number_of_samples = int(self.sample_rate / self.center_frequency)
        time_steps = np.linspace(0, (number_of_samples-1)*sample_time_interval, number_of_samples)
        freq_inteval = self.sample_rate / number_of_samples
        freq_steps = np.linspace(0, (number_of_samples - 1) * freq_inteval, number_of_samples)

        #  sine waves combined
        y = (1 * np.sin(2 * np.pi * start_freq * time_steps) +
             8 * np.sin(2 * np.pi  * mid_f1 * time_steps) +
             4 * np.sin(2 * np.pi * self.center_frequency * time_steps) +
             16 * np.sin(2 * np.pi * 10 * mid_f2 * time_steps) +
             24 * np.sin(2 * np.pi * end_freq * time_steps))

        self.plot_signal(time_steps, y, self.ax1)

        # fft windows

        x_signal = np.fft.fft(y)
        x_mag = np.abs(x_signal)
        x_mag = np.fft.fftshift(x_mag)
        f_plot = freq_steps[0:int(number_of_samples/2+1)]
        x_mag_plot = 2 * x_mag[0:int(number_of_samples/2+1)]
        x_mag_plot[0] = x_mag_plot[0] / 2
        self.plot_signal(f_plot, x_mag_plot, self.ax2)

        # fft - apply kaiser window

        # Generate a signal
        N = fft_size  # Number of samples
        fs = self.sample_rate  # Sampling frequency
        t = np.arange(N) / fs
        f = 10  # Frequency of the signal

        #  sine waves combined with fft size applied
        y_fft_sized = (1 * np.sin(2 * np.pi * start_freq * t) +
                       8 * np.sin(2 * np.pi * mid_f1 * t) +
                       4 * np.sin(2 * np.pi * self.center_frequency * t) +
                       16 * np.sin(2 * np.pi * 10 * mid_f2 * t) +
                       24 * np.sin(2 * np.pi * end_freq * t))

        # Apply the Kaiser window
        window = np.kaiser(N, beta=44)
        windowed_signal = y_fft_sized * window

        # Perform Fourier transform
        spectrum = np.fft.fft(windowed_signal)
        spectrum = np.fft.fftshift(spectrum)
        spectrum = np.abs(spectrum)
        freq = np.fft.fftfreq(N, 1 / fs)

        # Plot the spectrum

        self.plot_signal(freq, spectrum, self.ax3)


    def plot_signal(self, inteval, signal, ax):
        # create plot with labels and title
        ax.plot(inteval, signal, ".-", lw=0.4, c='blue')
        ax.set_title("Signal Sine Wave")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count (s)")
        ax.minorticks_on()
        ax.grid()
        ax.set_xlim(0, inteval[-1])
        plt.tight_layout()



if __name__ == '__main__':
    sig = Signal()
    sig.sine_wave(span=100e6, fft_size=512)
    plt.show()






