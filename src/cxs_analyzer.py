import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal.windows import flattop, hamming, kaiser, blackman, hann
from constants import Labels, Windows




class SdrSig():
    """
    Software defined radio signal.
    Used for testing purposes to simulate complex input signal.
    """

    def __init__(self, frequency_center=20e6, frequency_sample_rate=100e6, gain=0.2):
        self.center_freq = self.check_central_frequency_range(frequency_center)
        self.sample_rate = frequency_sample_rate
        self.gain = gain  # Increase or reduce this depending on the strength of the signals being studied.

    def frequency_generator_relative(self):
        """
        Generate frequencies which are fixed relative to the configured center frequency.
        Avoids generating freq which falls into 0th fft bin.
        :param self:
        :return: relative_freq_list:
        """

        f1 = self.sample_rate / 4
        f2 = 1
        f3 = -self.sample_rate / 4
        relative_freq_list = [f1, f2, f3]
        return relative_freq_list

    def frequency_sweep_generator_absolute(self):
        """
        Generate frequencies which are fixed in an absolute term, independent of the configured center frequency.
        It generates signals at each MHz.
        Generates frequency, even if it falls in the 0th fft bin/position.
        :return: list of absolute frequencies
        """
        frequency_step = 40e6  # Hz
        # calculating the start and the end freq using span - sample frequency/rate
        start_frequency = self.center_freq - self.sample_rate / 2
        end_frequency = self.center_freq + self.sample_rate / 2
        srf = int(np.ceil(start_frequency / frequency_step) * frequency_step)  # rounded value
        erf = int((end_frequency // frequency_step) * frequency_step) + 1
        freq_list = []
        for cur in range(srf, erf, int(frequency_step)):
            f_cur = self.center_freq - cur
            if f_cur == 0:
                f_cur = 0
            print("DBUG:absFreqs:{}={}".format(cur, f_cur))
            freq_list.append(f_cur)
        return freq_list

    def travers_samples(self, size):
        '''
        Actual rtlsdr returns iq data in complex notation.
        This currently returns real data
        '''
        gainMult = 10 ** (self.gain / 10)
        sample_time_interval = size / self.sample_rate
        t_start = random.random()
        self.t_times = np.linspace(t_start, int(t_start + sample_time_interval), int(self.sample_rate * sample_time_interval))
        signal_time_domain = []
        f = np.array(self.frequency_generator_relative())
        # f = np.array(self.frequency_sweep_generator_absolute())
        print(
            "INFO:testfft_rtlsdr: freqs [{}], sampRate [{}], tStart [{}], dur [{}], len [{}]".format(
                self.center_freq - f, self.sample_rate, t_start, sample_time_interval, len(self.t_times)))
        s = np.zeros(len(self.t_times), dtype=complex)
        for i in range(len(f)):
            sS = gainMult * np.sin(2 * np.pi * f[i] * self.t_times)
            sC = gainMult * np.cos(2 * np.pi * f[i] * self.t_times)
            signal_time_domain.append(sS + sC)
            s += signal_time_domain[i]
        return s

    def check_central_frequency_range(self, frequency):
        if -50e6 <= frequency <= 50e6:
            return frequency
        else:
            print("Center frequency = {0} Hz does not match requirement to be within -50Mhz to 50Mhz.".format(frequency))
            exit(1)

    def employ_window_fft(self, window_function, points):
        win = None
        if Windows.Bartlett in window_function:
            win = np.bartlett(points)
        elif Windows.Kaiser in window_function:
            win = np.kaiser(points, 15)
        elif Windows.Hamming in window_function:
            win = np.hamming(points)
        elif Windows.Hanning in window_function:
            win = np.hanning(points)
        elif Windows.Blackman in window_function:
            win = np.blackman(points)
        return win

    def test(self, signal_size, fft_size_points, fft_win_type, span):
        data_buffer = self.travers_samples(random.randint(1, signal_size)*2**16)
        window = self.employ_window_fft(fft_win_type, fft_size_points)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data_buffer), 1 / span)) + self.center_freq
        print("INFO:testfft_rtlsdr: freqs [{}] - [{}], fftSize [{}]".format(min(freqs), max(freqs), len(data_buffer)))
        fftCur = np.abs(np.fft.fft(data_buffer))/len(data_buffer)
        fftCur = np.fft.fftshift(fftCur)
        fftCurWin = np.abs(np.fft.fft(data_buffer*window))/len(data_buffer)
        fftCurWin = np.fft.fftshift(fftCurWin)
        plt.subplot(2,2,1)
        plt.plot(freqs, fftCur)
        plt.subplot(2,2,2)
        plt.plot(freqs, 10*np.log10(fftCur))
        plt.subplot(2,2,3)
        plt.plot(freqs, fftCurWin)
        plt.subplot(2,2,4)
        plt.plot(freqs, 10*np.log10(fftCurWin))
        plt.show()


if __name__ == '__main__':
    sdr = SdrSig(frequency_center=2e6)
    relative_freq_list = sdr.frequency_generator_relative()
    absolute_freq_list = sdr.frequency_sweep_generator_absolute()
    sdr.test(signal_size=2, fft_size_points=1024, fft_win_type=Windows.Hanning, span=6.5e6)

