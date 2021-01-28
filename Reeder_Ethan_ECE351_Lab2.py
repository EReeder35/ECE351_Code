# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #2                                 #
#   Due 2Feb2021                           #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})   # set font size

#   Part 1
steps = 1e-2    # define step size

t = np.arange(0, 10 + steps, steps)

y = np.cos(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y = cos(t)')
plt.title('Plotting Cosine Function')

t = np.arange(-5, 10 + steps, steps)

#   Part 2

def ramp(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
            
    return y

y = ramp(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('Ramp Function')
plt.xlabel('t')
plt.title('Plotting Ramp Function')


def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
            
    return y

y = step(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('Step Function')
plt.xlabel('t')
plt.title('Plotting Step Function')


def f(t):
    return (ramp(t) + 5*step(t-3) - ramp(t-3) - 2*step(t-6) - 2*ramp(t-6))

steps = 1e-3
t = np.arange(-5,10+steps,steps)

#   Part 3

# below code was modified for each of the tasks for Part 3, but the functions
# covered and shown in the attached graphs are
# y = f(-t)
# y = f(t-4)
# y = f(-t-4)
# y = f(t/2)
# y = f(2t)

y = f(t)

dt = np.diff(t)
dy = np.diff(y, axis=0)/dt

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y,'--',label = 'y(t)')
plt.plot(t[range(len(dy))],dy[:],label='dy(t)/dt')
plt.ylim([-2,10])
plt.grid()
plt.legend()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Plot for Lab 2')
