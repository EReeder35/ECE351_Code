# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #3                                 #
#   Due 30Mar2021                          #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})   # set font size

steps = 1e-2
t = np.arange(0,20+steps,steps)
T = 8

# Finding b_k terms

a = np.zeros((1501,1))     # 1500 arrays of length 1
for k in np.arange(1,1501):
    a[k] = 0

b = np.zeros((1501,1))     # 1500 arrays of length 1
for k in np.arange(1,1501):
    b[k] = 2/(k*np.pi) * (1 - np.cos(k*np.pi))
    
print('a_0: ', a[0])
print('a_1: ', a[1])
print('b_1: ', b[1])
print('b_2: ', b[2])
print('b_3: ', b[3])

    
# Summation Approximation
# uses N = {1, 3, 15, 50, 150, 1500}

# N = 1
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 1
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, total)
plt.grid()
plt.title('Fourier Series Approximations')
plt.xlabel('t')
plt.ylabel('N = 1')

# N = 3
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 3
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.subplot(3, 1, 2)
plt.plot(t, total)
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 3')    

# N = 15
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 15
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.subplot(3, 1, 3)
plt.plot(t, total)
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 15')   

# N = 50
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 50
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, total)
plt.grid()
plt.title('Fourier Series Approximations')
plt.xlabel('t')
plt.ylabel('N = 50')     

# N = 150
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 150
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.subplot(3, 1, 2)
plt.plot(t, total)
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 150')    

# N = 1500
total = 0   # reset since using same variable name (not absolutely necessary here)
N = 1500
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k * (2 * t * np.pi / T)))
plt.subplot(3, 1, 3)
plt.plot(t, total)
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 1500')                                