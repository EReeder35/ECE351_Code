# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #10                                #
#   Due 6April2021                         #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

plt.rcParams.update({'font.size': 14})   # set font size

steps = 100
omega = np.arange(1e3,1e6+steps,steps)

R = 1e3
L = 27e-3
C = 100e-9

# Part 1 Task 1

h_mag = (omega * L) / (np.sqrt((omega ** 4) * (R ** 2) * (L ** 2) * (C ** 2) + (omega ** 2)*(L ** 2 - 2 * (R ** 2) * L * C) + R ** 2))

h_ang = 90 - np.degrees(np.arctan2((omega * L) , (R - omega**2 * R * L * C)))

h_mag_DB = 20 * np.log10(h_mag)

plt.figure(figsize = (12, 8))
plt.title('Hand-Calculated')
plt.subplot(2,1,1)
plt.semilogx(omega, h_mag_DB)
plt.grid()
plt.xlim([1e3, 1e6])

plt.subplot(2,1,2)
plt.semilogx(omega, h_ang)
plt.grid()
plt.xlim([1e3, 1e6])
plt.ylim([-90,90])
plt.xlabel('Frequency - Part 1')

plt.show()

# Part 1 Task 2

num = [1/(R*C),0]
den = [1, 1/(R*C), 1/(L*C)]

w, mag, phase = sig.bode((num, den), omega)

plt.figure(figsize = (12,8))
plt.title('scipy.signal.bode Function')
plt.subplot(2,1,1)
plt.semilogx(w, mag)
plt.grid()
plt.xlim([1e3, 1e6])

plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.grid()
plt.xlim([1e3, 1e6])
plt.ylim([-90,90])
plt.xlabel('Frequency - Part 2')

plt.show()

# Part 1 Task 3

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega, dB = True, Hz = True, deg = True, plot = True)

# Part 2 Task 1
fs = 2 * np.pi * 50000
t_steps = 1/fs

t = np.arange(0,0.01+t_steps,t_steps)
x = np.cos(2 * np.pi * 100 * t) + np.cos(2 * np.pi * 3024 * t) + np.sin(2 * np.pi * 50000 * t)

plt.figure(figsize = (12,8))
plt.grid()
plt.plot(t, x)

z, p = sig.bilinear(num, den, fs) # turns h(s) into a z domain function

y = sig.lfilter(z, p, x) # runs x through the z domain version of h(s)

plt.figure(figsize = (12,8))
plt.grid()
plt.plot(t, y)