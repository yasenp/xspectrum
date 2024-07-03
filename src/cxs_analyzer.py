import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from constants import Labels


class SdrSig():
    """
    Software defined radio signal.
    Used for testing purposes to simulate complex input signal.
    """

    def __init__(self, frequency_center=42e6, frequency_sample_rate=100e6, gain=0.5):
        self.center_freq = frequency_center
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

    def frequency_generator_absolute(self):
        """
        Generate frequencies which are fixed in an absolute term, independent of the configured center frequency.
        It generates signals at each MHz.
        Generates frequency, even if it falls in the 0th fft bin/position.
        :return: list of absolute frequencies
        """
        frequency_step = 35e6  # Hz
        start_frequency = self.center_freq - self.sample_rate / 2
        end_frequency = self.center_freq + self.sample_rate / 2
        sr = int(np.ceil(start_frequency / frequency_step) * frequency_step)  # rounded value
        er = int((end_frequency // frequency_step) * frequency_step) + 1
        freq_list = []
        for cur in range(sr, er, int(frequency_step)):
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
        sample_duration = size / self.sample_rate
        t_start = random.random()
        self.t_times = np.linspace(t_start, int(t_start + sample_duration), int(self.sample_rate * sample_duration))
        signal_time_domain = []
        # f = np.array(self.frequency_generator_relative())
        f = np.array(self.frequency_generator_absolute())
        print(
            "INFO:testfft_rtlsdr: freqs [{}], sampRate [{}], tStart [{}], dur [{}], len [{}]".format(
                self.center_freq - f, self.sample_rate, t_start, sample_duration, len(self.t_times)))
        s = np.zeros(len(self.t_times), dtype=complex)
        for i in range(len(f)):
            sS = gainMult * np.sin(2 * np.pi * f[i] * self.t_times)
            sC = gainMult * np.cos(2 * np.pi * f[i] * self.t_times)
            signal_time_domain.append(sS + sC * 1j)
            s += signal_time_domain[i]
        return s


    def test(self):
        data_buffer = self.travers_samples(random.randint(1,10)*2**16)
        freqs = np.linspace(0, len(data_buffer) * 1/self.sample_rate * int(self.sample_rate / self.center_freq))
        print(freqs)
        win = np.kaiser(len(data_buffer), 15)
        win = np.hanning(len(data_buffer))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data_buffer), 1 / self.sample_rate)) + self.center_freq
        print("INFO:testfft_rtlsdr: freqs [{}] - [{}], fftSize [{}]".format(min(freqs), max(freqs), len(data_buffer)))
        fftCur = np.abs(np.fft.fft(data_buffer)) / len(data_buffer)
        fftCur = np.fft.fftshift(fftCur)
        fftCurWin = np.abs(np.fft.fft(data_buffer * win)) / len(data_buffer)
        fftCurWin = np.fft.fftshift(fftCurWin)
        plt.subplot(2, 2, 1)
        plt.plot(freqs, fftCur)
        plt.subplot(2, 2, 2)
        plt.plot(freqs, 10 * np.log10(fftCur))
        plt.subplot(2, 2, 3)
        plt.plot(freqs, fftCurWin)
        plt.subplot(2, 2, 4)
        plt.plot(freqs, 10 * np.log10(fftCurWin))
        plt.show()




if __name__ == '__main__':
    sdr = SdrSig()
    relative_freq_list = sdr.frequency_generator_relative()
    absolute_freq_list = sdr.frequency_generator_absolute()
    sdr.test()

