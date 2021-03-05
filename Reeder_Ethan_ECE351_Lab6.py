# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #6                                 #
#   Due 9Mar2021                           #
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

steps = 1e-3

t = np.arange(0,2+steps,steps)

## PART 1

# Task 1

y = (0.5 + np.exp(-6*t) - 0.5*np.exp(-4*t)) * step(t)

plt.figure(figsize = (10,7))

# Task 2

num = [1,6,12]
den = [1,10,24]

tstep, ystep = sig.step((num, den), T = t)

plt.subplot(2,1,1)
plt.plot(t, y)
plt.ylabel('Prelab-Derived Step Response')
plt.grid(which='both')

plt.subplot(2,1,2)
plt.plot(tstep,ystep)
plt.ylabel('Step Function')
plt.grid(which='both')

plt.show()

# Task 3
den2 = [1,10,24,0]

res1 = sig.residue(num,den2)

print('Part 1 R, P, and K Values:')
print('R: ', res1[0]) # r value
print('P: ', res1[1]) # p value
print('K: ', res1[2]) # k value

## PART 2

t2 = np.arange(0,4.5+steps,steps)

# Task 1

# h2 = (25250)/(s^5 + 18s^4 + 218s^3 + 2036s^2 + 9085s + 25250)
num3 = [25250]
den3 = [1,18,218,2036,9085,25250,0]

res2 = sig.residue(num3,den3)

print('Part 2 R, P, and K Values:')
print('R: ', res2[0]) # r value
print('P: ', res2[1]) # p value
print('K: ', res2[2]) # k value

def cos_method(z,t):
    y = 0
    for i in range(len(z[0])):
        k_mag = np.abs(z[0][i])
        k_phase = np.angle(z[0][i])
        a = np.real(z[1][i])
        w = np.imag(z[1][i])
        y = y + (k_mag * np.exp(a*t) * np.cos(w*t + k_phase)) * step(t)
    return y

step_cos = cos_method(res2, t2)

# Task 3
num4 = [25250]
den4 = [1,18,218,2036,9085,25250]

tstep2, ystep2 = sig.step((num4, den4), T = t2)

plt.subplot(2,1,1)
plt.plot(t2, step_cos)
plt.ylabel('Cos Method')
plt.grid(which='both')

plt.subplot(2,1,2)
plt.plot(tstep2,ystep2)
plt.ylabel('Step Function')
plt.grid(which='both')

plt.show()