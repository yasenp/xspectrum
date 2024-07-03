import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


class Signal(object):
    # signal constants
    def __init__(self):
        self.xt_sampled = None
        self.t_min = int(-50e6)
        self.t_max = int(50e6)
        self.num_t = 10000
        self.t = np.linspace(self.t_min, self.t_max, self.num_t)
        self.freq = 25e6
        self.xt = np.sin(2*np.pi*self.freq*self.t)
        self.mat = plt

    def init_plot(self):
        self.mat.plot(self.t, self.xt)
        self.mat.xlabel("Time")
        self.mat.ylabel("Amplitude")
        self.mat.title("Plot of sin wave")

    def init_plot_sampling(self):
        # Defining sampling period / sampling frequency
        self.fs = 5
        self.ts = 1/self.fs
        # create pulse train for sampling, use linspace or arrange. (What is the difference?)
        self.pulse_train = np.arange(self.t_min, self.t_max, self.ts)
        # Plot the pulse train
        # self.mat.stem(self.pulse_train, np.ones(len(self.pulse_train)))

    def init_plot_quant(self):
        self.init_plot_sampling()
        # Show quantization
        self.xt_sampled = np.sin(2*np.pi*self.freq*self.pulse_train)
        self.mat.stem(self.pulse_train, self.xt_sampled)

    def show_plot(self):
        self.mat.show()


if __name__ == '__main__':
    sine_wave = Signal()
    sine_wave.init_plot()
    # sine_wave.init_plot_quant()
    sine_wave.show_plot()



