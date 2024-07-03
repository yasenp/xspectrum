import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import struct
from scipy.fftpack import fft
import time


class Signal(object):
    def __init__(self):

        # stream constants
        self.fig = None
        self.CHUNK = int(100e6)
        self.FORMAT = pyaudio.paInt24
        self.CHANNELS = 2
        self.RATE = int(1000)
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        self.init_plots()
        # self.start_plot()

    def init_plots(self):

        # x variables for plotting
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # create a line object with random data
        self.line, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            xf, np.random.rand(self.CHUNK), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.CHUNK, 2 * self.CHUNK],
        )
        plt.setp(ax2, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_xlim(20, self.RATE / 2)

        # show axes
        thismanager = plt.get_current_fig_manager()
        # thismanager.setGeometry(5, 120, 1910, 1070)
        plt.show(block=True)

    def on_click(self, event):
        self.pause = True

if __name__ == '__main__':
    Signal()
