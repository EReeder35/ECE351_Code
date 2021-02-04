# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #3                                 #
#   Due 9Feb2021                           #
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

## PART 2 TASK 1

def f1(t):
    return (step(t-2) - step(t-9))
    
def f2(t):
    return (np.exp(-t) * step(t))

def f3(t):
    return (ramp(t-2)*(step(t-2) - step(t-3)) + ramp(4-t)*(step(t-3) - step(t-4)))

steps = 1e-2
t = np.arange(0,20+steps,steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN-1], steps)

x = f1(t)
y = f2(t)
z = f3(t)

## PART 2 TASK 2

plt.figure(figsize = (10,7))

plt.subplot(3,1,1)
plt.plot(t,x)
plt.ylabel('f1(t)')
plt.grid(which='both')

plt.subplot(3,1,2)
plt.plot(t,y)
plt.ylabel('f2(t)')
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,z)
plt.ylabel('f3(t)')
plt.xlabel('t')
plt.grid(which='both')

plt.show()

## PART 3 TASK 1

def convolute(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 -1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 -1)))
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
        
## PART 2 TASK 2
# Convolve f1 and f2

convolve12 = convolute(x, y)*steps
convolve12check = sig.convolve(x, y)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolve12, label = 'Defined Convolution')
plt.plot(tExtended, convolve12check, '--', label = 'scipy.signal Convolution')
plt.ylim([0, 1.2])
plt.xlim([0, 20])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('f_1(t) * f_2(t)')
plt.title('Convolution of f1 and f2')
plt.show()

## PART 2 TASK 3
# Convolve f2 and f3

convolve23 = convolute(y, z)*steps
convolve23check = sig.convolve(y, z)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolve23, label = 'Defined Convolution')
plt.plot(tExtended, convolve23check, '--', label = 'scipy.signal Convolution')
plt.ylim([0, 0.8])
plt.xlim([0, 20])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('f_2(t) * f_3(t)')
plt.title('Convolution of f2 and f3')
plt.show()


## PART 2 TASK 4
# Convolve f1 and f3

convolve13 = convolute(x, z)*steps
convolve13check = sig.convolve(x, z)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, convolve13, label = 'Defined Convolution')
plt.plot(tExtended, convolve13check, '--', label = 'scipy.signal Convolution')
plt.ylim([0, 1.2])
plt.xlim([0, 20])
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('f_1(t) * f_3(t)')
plt.title('Convolution of f1 and f3')
plt.show()
