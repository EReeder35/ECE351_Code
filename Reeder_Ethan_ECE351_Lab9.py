# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #9                                 #
#   Due 6Apr2021                           #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fftp

steps = 1e-2
t = np.arange(0,2,steps)

def fft(f, fs):
    N = len(x)
    x_fft = fftp.fft(x) # perform fast fourier transform
    x_fft_shifted = fftp.fftshift(x_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    x_mag = np.abs(x_fft_shifted)/N
    x_phi = np.angle(x_fft_shifted)

    return freq, x_mag, x_phi

def fft_simple(f, fs):
    N = len(x)
    x_fft = fftp.fft(x) # perform fast fourier transform
    x_fft_shifted = fftp.fftshift(x_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    x_mag = np.abs(x_fft_shifted)/N
    x_phi = np.angle(x_fft_shifted)
    
    for i in range(0, N-1):
        if x_mag[i] < 1e-10:
            x_phi[i] = 0

    return freq, x_mag, x_phi

# Finding b_k terms for Fourier Series

b = np.zeros((16,1))     # 1500 arrays of length 1
for k in np.arange(1,16):
    b[k] = 2/(k*np.pi) * (1 - np.cos(k*np.pi))

# Plotting Function
def fft_plot(t, x, freq, x_mag, x_phi, title):
    
    # original function vs t
    plt.figure(figsize = (12, 8))
    plt.subplot(3,1,1)
    plt.plot(t, x)
    plt.grid()
    plt.title(title)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    
    # full t range
    plt.subplot(3,2,3)
    plt.stem(omega, mag)
    plt.grid()
    plt.ylabel('|X(f)|')
    
    plt.subplot(3,2,4)
    plt.stem(omega, mag)
    plt.grid()
    plt.xlim([-2,2])
    
    # Limiting t values to only "relevant" terms
    plt.subplot(3,2,5)
    plt.stem(omega, phi)
    plt.grid()
    plt.ylabel('/_ X(f)')
    plt.xlabel('f[Hz]')
    
    plt.subplot(3,2,6)
    plt.stem(omega, phi)
    plt.grid()
    plt.xlim([-2,2])
    plt.xlabel('f[Hz]')
    
    plt.tight_layout()
    plt.show()

    return 0

def fft_plot_simple(t, x, freq, x_mag, x_phi, title):
    
    # original function vs t
    plt.figure(figsize = (12, 8))
    plt.subplot(3,1,1)
    plt.plot(t, x)
    plt.grid()
    plt.title(title)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    
    # full t range
    plt.subplot(3,2,3)
    plt.stem(omega, mag)
    plt.grid()
    plt.ylabel('|X(f)|')
    
    plt.subplot(3,2,4)
    plt.stem(omega, mag)
    plt.grid()
    plt.xlim([-15,15])
    
    # Limiting t values to only "relevant" terms
    plt.subplot(3,2,5)
    plt.stem(omega, phi)
    plt.grid()
    plt.ylabel('/_ X(f)')
    plt.xlabel('f[Hz]')
    
    plt.subplot(3,2,6)
    plt.stem(omega, phi)
    plt.grid()
    plt.xlim([-15,15])
    plt.xlabel('f[Hz]')
    
    plt.tight_layout()
    plt.show()

    return 0

fs = 100
steps = 1/fs
t = np.arange(0,2,steps)

# Task 1
x = np.cos(2*np.pi*t)
#fs = 100

title = 'Task 1 - User Defined FFT'

omega, mag, phi = fft(x, fs)
fft_plot(t, x, omega, mag, phi, title)

# Task 2
x = 5 * np.sin(2*np.pi*t)
fs = 100
title = 'Task 2 - User Defined FFT'

omega, mag, phi = fft(x, fs)
fft_plot(t, x, omega, mag, phi, title)

# Task 3
x = 2 * np.cos((4*np.pi*t) - 2) + (np.sin((12*np.pi*t) + 3)) ** 2
fs = 100
title = 'Task 3 - User Defined FFT'

omega, mag, phi = fft(x, fs)
fft_plot(t, x, omega, mag, phi, title)

# Task 4.1
x = np.cos(2*np.pi*t)
fs = 100
title = 'Task 4.1 - User Defined FFT'

omega, mag, phi = fft_simple(x, fs)
fft_plot_simple(t, x, omega, mag, phi, title)

# Task 4.2
x = 5 * np.sin(2*np.pi*t)
fs = 100
title = 'Task 4.2 - User Defined FFT'

omega, mag, phi = fft_simple(x, fs)
fft_plot_simple(t, x, omega, mag, phi, title)

# Task 4.3
x = 2 * np.cos((4*np.pi*t) - 2) + (np.sin((12*np.pi*t) + 3)) ** 2
fs = 100
title = 'Task 4.3 - User Defined FFT'

omega, mag, phi = fft_simple(x, fs)
fft_plot_simple(t, x, omega, mag, phi, title)

# Task 5
t = np.arange(0,16,steps)
T = 8

# N = 15
total = 0
N = 15
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))

fs = 100
title = 'Task 5 - Fourier Series through FFT'

omega, mag, phi = fft_simple(total, fs)
fft_plot_simple(t, total, omega, mag, phi, title)
