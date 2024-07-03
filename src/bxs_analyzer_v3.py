import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from constants import Labels


class Signal(object):
    # signal constants
    def __init__(self, span):
        self.center_frequency = span/2
        if -50e6 <= self.center_frequency <= 50e6:
            self.input_signal_frequency = self.center_frequency
        else:
            print("Center frequency does not match requirement.")
            exit(1)

        self.span = span

        # sin for plot
        self.signal = None
        self.signal_sampled = None

        # define sampling period / sampling frequency
        self.sampling_rate = int(100e6)  # Hz
        self.sample_time_interval = 1/self.sampling_rate
        self.num_samples = int(self.sampling_rate / self.input_signal_frequency)
        self.num_samples_span = int(self.span / self.input_signal_frequency)
        self.frequency_interval = self.span/self.num_samples_span

        # create time steps of time complex signal
        self.time_steps = np.linspace(0, (self.num_samples - 1) * self.sample_time_interval, self.num_samples)
        f_start = self.center_frequency - span/2
        f_end = self.center_frequency + span / 2
        #  create frequency steps x-axis for frequency spectrum
        self.frequency_step = np.linspace(f_start, f_end, self.num_samples_span)

        # create figure ax 1 and 2
        self.fig, [self.ax1, self.ax2, self.ax3] = plt.subplots(nrows=3, ncols=1)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(8)

    def create_plot(self, time_inteval, signal, ax, config):
        # create plot with labels and title
        ax.plot(time_inteval, signal, "-", lw=0.4, c='black')
        ax.set_title(config['Title'])
        ax.set_xlabel(config['Time'])
        ax.set_ylabel(config['Count'])
        ax.minorticks_on()
        ax.grid()
        ax.set_xlim(0, time_inteval[-1])
        plt.tight_layout()

    def init_plot_signal(self):
        ax_config = {'Title': Labels.time_signal_title, 'Time': Labels.time_in_sec, 'Count': Labels.counts_in_sec}
        # create sine wave signal
        self.signal = 50 * np.sin(2 * np.pi * self.input_signal_frequency * self.time_steps)
        # create plot with labels and title
        self.create_plot(self.time_steps, self.signal, self.ax1, ax_config)

    def init_plot_fft(self):
        ax_config = {'Title': Labels.time_signal_fft_title, 'Time': Labels.frequency,
                     'Count': Labels.amplitude}
        # convert signal to series of complex numbers
        X = fft(self.signal)

        # Magnitude of fft output normalized to numbers of samples
        X_mag = np.abs(X)/ self.num_samples_span

        # Only plotting half of sampling frequency (just positive frequencies)
        f_plot = self.frequency_step[0:int(self.num_samples_span/2 + 1)]

        # Get Magnitude
        X_mag_plot = 2 * X_mag[0:int(self.num_samples_span/2 + 1)]
        X_mag_plot[0] = X_mag_plot[0] / 2

        # create plot with labels and title
        self.create_plot(f_plot, X_mag_plot, self.ax2, ax_config)

    def init_plot_quantization(self):
        ax_config = {'Title': Labels.time_signal_fft_title, 'Time': Labels.frequency,
                     'Count': Labels.amplitude}
        # Show quantization
        self.signal_quant = np.sin(2 * np.pi * self.input_signal_frequency * self.frequency_step)
        self.create_plot(self.frequency_step, self.signal_quant, self.ax3, ax_config)

    def show_plots(self):
        plt.show()


if __name__ == '__main__':

    sine_wave = Signal(100e6)
    sine_wave.init_plot_signal()
    sine_wave.init_plot_fft()
    sine_wave.init_plot_quantization()
    sine_wave.show_plots()
