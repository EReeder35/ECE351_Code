# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #5                                 #
#   Due 2Mar2021                           #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})   # set font size

## STEP FUNCTION
def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
            
    return y

## PART 1

steps = 1e-5

t = np.arange(0,1.2e-3+steps,steps)

# Hand-Derived Plot

R = 1e3
L = 27e-3
C = 100e-9

def sin_method(b, c, d, f):   # where b, c, d, f are the coefficients of the num and den respectively
    a = -d/2
    w = 0.5 * np.sqrt(d**2 - 4 * f + 0j) + 0j
    p = a + w + 0j
    g = (b * (p+0j) + c + 0j)
    g_mag = np.abs(g)
    g_phase = np.angle(g)
    output = ((g_mag / abs(w)) * np.exp(a*t) * np.sin(abs(w)*t + g_phase) * step(t))
    return output

f = sin_method((1/(R*C)), 0, (1/(R*C)), (1/(L*C)))

# Python Calculated

num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

tout, yout = sig.impulse((num, den), T = t)

plt.figure(figsize = (10,7))

plt.subplot(2,1,1)
plt.plot(tout, yout)
plt.ylabel('Laplace Function in Python')
plt.grid(which='both')

plt.subplot(2,1,2)
plt.plot(t,f)
plt.ylabel('Prelab Graph')
plt.grid(which='both')

plt.show()

## PART 2
tstep, ystep = sig.step((num, den), T = t)

plt.figure(figsize = (10,7))

plt.plot(tstep, ystep)
plt.ylabel('Step Response')
plt.grid(which='both')

plt.show()