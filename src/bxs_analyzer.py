"""
    Notebook for streaming data from a microphone in realtime

    audio is captured using pyaudio
    then converted from binary data to ints using struct
    then displayed using matplotlib

    scipy.fftpack computes the FFT

    if you don't have pyaudio, then run

    # >>> pip install pyaudio

    note: with 2048 samples per chunk, I'm getting 20FPS
    when also running the spectrum, its about 15FPS
"""
import matplotlib.pyplot as plt
import numpy as np
import struct
from scipy.fftpack import fft
import time


class SignalStream(object):
    def __init__(self):

            # stream constants

        # Sampling frequency (Hz). Can reconstruct signal to half of this frequency
        # This is also sampling rate it is  equal to average number of samples obtained in one second 1/T Step.
        self.fs = 300e6
        # sample time interval
        self.t_step = 1 / self.fs
        # input signal frequency (Hz)
        self.input_signal_frequency = 100e6
        # number of samples. Ex.: Use int(10*Fs/F0) to increase number of samples (10 cycles of samples)
        self.sampling_rate = int(100 * self.fs / self.input_signal_frequency)
        # Create time steps of time domain signal
        self.time_steps = np.linspace(-50, 50)
        # freq interval for each frequency bin (sampling frequency divided by number of samples)
        self.f_step = self.fs / self.sampling_rate
        # Create freq steps –> x-axis for frequency spectrum
        self.freq_time_steps = np.linspace(0, (self.sampling_rate - 1) * self.f_step, self.sampling_rate)
        # Create a 1 sine signal 100 MHz time domain signal and another 0.5
        self.signal = 1 * np.cos(2 * np.pi * self.input_signal_frequency * self.time_steps)

        self.fig = None
        self.f_plot = None
        self.X_mag_plot = None
        self.X_mag = None
        self.init_plots()

    # def init_plots(self):
    #
    #     # x variables for plotting
    #
    #     # create matplotlib figure and axes
    #     self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
    #
    #     # Plot frequency of spectrum using matplotlib
    #     self.fig.set_figheight(8)
    #     self.fig.set_figwidth(8)
    #     self.fft_perform()
    #
    #     ax1.plot(self.time_steps, self.signal, '-', lw=0.4, c ='black')
    #
    #     ax2.plot(self.f_plot, self.X_mag_plot, '-', lw=0.4, c ='black')
    #
    #     # format waveform axes 1
    #     ax1.set_title('Complex Input Signal')
    #     ax2.set_title('Magnitute of Spectrum')
    #     ax1.set_xlabel('Time(s)')
    #     ax1.set_ylabel('Count(s)')
    #     ax1.minorticks_on()
    #
    #     # format waveform axes 2
    #
    #     ax2.set_xlabel('Frequency (Hz)')
    #     ax2.set_ylabel('Amplitude (pk)')
    #
    #
    #     ax1.grid()
    #     ax2.grid()
    #     ax1.set_xlim(0, self.time_steps[-1])
    #     ax2.set_xlim(0, self.f_plot[-1])
    #     plt.tight_layout()
    #     plt.show()
    #
    # def fft_perform(self):
    #     # Step 2: Perform FFT using SciPy ‘fft’ function
    #     X = fft(self.signal)  # X is a series of complex numbers
    #     self.X_mag = np.abs(X) / self.sampling_rate  # Magnitude of fft output normalized to number of samples
    #     self.f_plot = self.freq_time_steps[0:int(self.sampling_rate / 2 + 1)]  # Only plotting half of sampling frequency (just positive frequencies)
    #     self.X_mag_plot = 2 * self.X_mag[0:int(self.sampling_rate / 2 + 1)]  # Get Magnitude
    #     # Get correct DC Component (does not require multiplication by 2) and does not require taking log.
    #     self.X_mag_plot[0] = self.X_mag_plot[0] / 2  # Compute DC



    def init_plots(self):

        # x variables for plotting
        x = np.arange(0, 50 * self.fs, self.fs)
        xf = np.linspace(0, self.f_step, self.sampling_rate)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create a line object with random data
        self.line, = ax1.plot(x, np.random.rand(self.sampling_rate), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            xf, np.random.rand(self.sampling_rate), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.sampling_rate)
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.sampling_rate, 2 * self.sampling_rate],
        )
        plt.setp(ax2, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_xlim(20, self.sampling_rate / 2)

        # show axes
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(5, 120, 1910, 1070)
        plt.show(block=False)

    def start_plot(self):

        print('stream started')
        frame_count = 0
        start_time = time.time()

        while not self.pause:
            data = self.stream.read(self.sampling_rate)
            data_int = struct.unpack(str(2 * self.sampling_rate) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128

            self.line.set_ydata(data_np)

            # compute FFT and update line
            yf = fft(data_int)
            self.line_fft.set_ydata(
                np.abs(yf[0:self.sampling_rate]) / (128 * self.sampling_rate))

            # update figure canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.signal)

    def onClick(self, event):
        self.pause = True

if __name__ == '__main__':
    SignalStream()
