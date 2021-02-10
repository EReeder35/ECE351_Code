# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #4                                 #
#   Due 16Feb2021                          #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})   # set font size

## RAMP FUNCTION
def ramp(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
            
    return y

## STEP FUNCTION
def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
            
    return y

## CONVOLUTION FUNCTION
def convolute(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 -2)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 -2)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if(i - j + 1 > 0):
                try:
                    result[i] += f1Extended[j]*f2Extended[i - j + 1]
                except:
                    print(i , j)
    
    return result

## PART 1 TASK 1

def h1(t):
    return np.exp(2*t) * step(1-t)

def h2(t):
    return step(t-2) - step(t-6)

f0 = 0.25

def h3(t):
    return np.cos(2 * np.pi * f0 * t) * step(t)

steps = 1e-2
t = np.arange(-10,10+steps,steps)
NN = len(t)
tExtended = np.arange(2 * t[0], 2*t[NN-1], steps)

h1 = h1(t)
h2 = h2(t)
h3 = h3(t)

## PART 1 TASK 2
"""

plt.figure(figsize = (10,7))

plt.subplot(3,1,1)
plt.plot(t,h1)
plt.ylabel('h1(t)')
plt.grid(which='both')

plt.subplot(3,1,2)
plt.plot(t,h2)
plt.ylabel('h2(t)')
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,h3)
plt.ylabel('h3(t)')
plt.xlabel('t')
plt.grid(which='both')

plt.show()
"""

## PART 2 TASK 1

convolveh1u = convolute(h1, step(t))*steps
def h1hand(t):
    return 0.5 *(( np.exp(2*t) * step(1-t)) + (np.exp(2) * step(t - 1)))
h1h = h1hand(t)
##y1 = 0.5 * np.exp(2*t) * step(1-t) + np.exp(2) * step(t-1)

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolveh1u, label = 'Defined Convolution')
plt.plot(t, h1h, '--', label = 'Hand Calculated Convolution')
plt.ylim([-5,8])
plt.xlim([-10,15])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('h_1(t) * u(t)')
plt.title('Step Response of h_1(t)')
plt.show()

convolveh2u = convolute(h2, step(t))*steps
def h2hand(t):
    return ((t-2) * step(t-2)) - ((t-6) * step(t-6))
h2h = h2hand(t)

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolveh2u, label = 'Defined Convolution')
plt.plot(t, h2h, '--', label = 'Hand Calculated Convolution')
plt.ylim([-1,5])
plt.xlim([-10,20])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('h_2(t) * u(t)')
plt.title('Step Response of h_2(t)')
plt.show()

convolveh3u = convolute(h3, step(t))*steps
def h3hand(t):
    return (1 / (2 * np.pi * f0)) * np.sin(2 * np.pi * f0 * t) * step(t)
h3h = h3hand(t)

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolveh3u, label = 'Defined Convolution')
plt.plot(t, h3h, '--', label = 'Hand Calculated Convolution')
plt.ylim([-2,2])
plt.xlim([-10,20])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('h_3(t) * u(t)')
plt.title('Step Response of h_3(t)')
plt.show()
