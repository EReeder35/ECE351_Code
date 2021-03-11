# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #7                                 #
#   Due 23Mar2021                          #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})   # set font size

steps = 1e-3

t = np.arange(0,10+steps,steps)

numG = [1, 9]
denG = [1, -2, -40, -64]
numA = [1, 4]
denA = [1, 4, 3]
numB = [1, 26, 168]
denB = [1]

## PART 1
# POLES AND ZEROES

outG = sig.tf2zpk(numG, denG)
outA = sig.tf2zpk(numA, denA)
outB = sig.tf2zpk(numB, denB)

print('G: ', outG)
print('A: ', outA)
print('B: ', outB)

# OPEN LOOP TRANSFER FUNCTION

num3 = [1, 9]
den1 = sig.convolve([1, 1], [1, 2])
den2 = sig.convolve([1, 3], [1, -8])
den3 = sig.convolve(den1, den2)
print('Numerator: ', num3)
print('Denominator: ', den3)

# PLOT

tstep1, ystep1 = sig.step((num3, den3), T = t)

plt.plot(tstep1,ystep1)
plt.ylabel('Part 1 Step Response')
plt.grid(which='both')

plt.show()

## PART 2
# POLES AND ZEROES

numH = sig.convolve(numA, numG)
denH1 = sig.convolve(numB, numG)
denH2 = denG + denH1
denH = sig.convolve(denA, denH2)

outH = sig.tf2zpk(numH, denH)

print('CL Numerator: ', numH)
print('CL Denominator: ', denH)

print('H Zeroes: ', outH[0])
print('H Poles: ', outH[1])
print('H Gain: ', outH[2])

# PLOT

tstep2, ystep2 = sig.step((numH, denH), T = t)

plt.plot(tstep2,ystep2)
plt.ylabel('Part 2 Step Response')
plt.grid(which='both')

plt.show()

# Test for Q1

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


denTest = convolute([1, 1], [1, 2])
print('scipy.convolve: ', den1)
print('User Defined: ', denTest)