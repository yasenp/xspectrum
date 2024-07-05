# xspectrum
# Foobar

xspectrum is a Python script for dealing with signal wave spectrum analysis.

## Python

Version 3.11.6

## Installation

Clone the project to your local workstation.

## Bash

```bash
python xs_analyzer.py 
```
## Usage

```python
# create instance of the signal and define the center frequency

sig = Signal(center_frequency=2.4e6)

# create Sine Wave as sweep parameter is False
sig.waveform(span=100e6, fft_size=4096, window_type=Windows.Kaiser, sweep=False)

# create Frequency Sweep Wave as sweep parameter is True
sig.waveform(span=100e6, fft_size=4096, window_type=Windows.Kaiser, sweep=True)

# Control span and fft_size (proportional affects resolution)
sig.waveform(span=100e6, fft_size=4096, window_type=Windows.Kaiser, sweep=False)

# Define windows type using windows_type function
sig.waveform(span=100e6, fft_size=4096, window_type=Windows.Kaiser, sweep=False)

#Supported window functions:

    Windows.Hanning
    Windows.Hamming
    Windows.Kaiser
    Windows.Blackman
    Windows.Bartlett

# plot all figures
plt.show()
```

## Contributing

https://www.youtube.com/watch?v=wqLEqJSq4AA
https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
https://dsp.stackexchange.com/questions/11312/why-should-one-use-windowing-functions-for-fft
https://support.ircam.fr/docs/AudioSculpt/3.0/co/FFT%20Size.html#:~:text=By%20default%2C%20the%20FFT%20size,proportionnaly%20with%20the%20oversampling%20factor.&text=If%20the%20window%20size%20is,the%20closest%20power%20of%20two.

# Examples

Fugure 1

![figure1.png](images%2Ffigure1.png)

Figure 2

![figure2.png](images%2Ffigure2.png)
 Figure 3

![figure3.png](images%2Ffigure3.png)

#