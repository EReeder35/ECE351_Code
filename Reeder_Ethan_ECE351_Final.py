# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab Final                              #
#   Due 8May2021                           #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
import scipy.fftpack as fftp
import control as con

# stem signal workaround
def make_stem(ax, x, y, color = 'k', style = 'solid', label = '', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], color = 'r')
    ax.vlines(x, 0, y, color = color, linestyles = style, label = label, linewidths = linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])

# simple fast fourier transform
def fft_simple(f, fs, low_lim, high_lim):
    N = len(f)
    x_fft = fftp.fft(f) # perform fast fourier transform
    x_fft_shifted = fftp.fftshift(x_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    x_mag = np.abs(x_fft_shifted)/N
    x_phi = np.angle(x_fft_shifted)
    
    for i in range(0, N-1):
        if x_mag[i] < 1e-10:
            x_phi[i] = 0
    
    fig, ax1 = plt.subplots(figsize = (10,7))
    
    #plt.subplot(ax1)
    plt.xscale('log')
    plt.xlim([low_lim, high_lim])
    make_stem(ax1, freq, x_mag)
    plt.grid(which='both')
    plt.show()
    
    return 0

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values
"""
plt.figure(figsize = (10,7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noise Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [v]')
plt.show()


# noise magnitudes and corresponding frequencies for whole frequency range
fs = 1e6
fft_simple(sensor_sig, fs, 0, 1e6)

# below band
fft_simple(sensor_sig, fs, 0, 1.8e3)

# above band
fft_simple(sensor_sig, fs, 2e3, 1e6)

# in band
fft_simple(sensor_sig, fs, 1.8e3, 2e3)
"""
# filter design
steps = 100

R = 5e2
L = 27e-3
C = 260e-9

num = [1/(R*C),0]
den = [1, 1/(R*C), 1/(L*C)]

sys = con.TransferFunction(num, den)

"""
# position measurement attenuated by less than -0.3 dB
omega1 = np.arange(6283.19,15707+steps,steps)
_ = con.bode(sys, omega1, dB = True, Hz = True, deg = True, plot = True)


# low-frequency vibration attenuated by at least -30 dB
omega2 = np.arange(0,6283.19+steps,steps)
_ = con.bode(sys, omega2, dB = True, Hz = True, deg = True, plot = True)



# low-frequency vibration attenuated by at least -30 dB
omega3 = np.arange(15707,628300+steps,steps)
_ = con.bode(sys, omega3, dB = True, Hz = True, deg = True, plot = True)



omega4 = np.arange(628300,1e9+steps,steps)
_ = con.bode(sys, omega4, dB = True, Hz = True, deg = True, plot = True)
"""

fs = 1e6

# running the input signal through the filter
z, p = sig.bilinear(num, den, fs) # turns h(s) into a z domain function

y = sig.lfilter(z, p, sensor_sig) # runs x through the z domain version of h(s)

# whole frequency range
fft_simple(y, fs, 0, 1e6)

# below band
fft_simple(y, fs, 0, 1.8e3)

# above band
fft_simple(y, fs, 2e3, 1e6)

# in band
fft_simple(y, fs, 1.8e3, 2e3)

plt.figure(figsize = (10,7))
plt.plot(t, y)
plt.grid()
plt.title('Clean Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [v]')
plt.show()
