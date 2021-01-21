# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #1                                 #
#   Due 26Jan2021                          #
#                                          #
############################################


# main libraries to be used for this lab

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import pandas as pd
import time
from scipy.fftpack import fft, fftshift


t = 1
print(t)
print("t =", t)
print('t =', t, "seconds")
print('t is now =', t/3, '\n... and can be rounded using round()', round(t/3,4))

print (3**2)

list1 = [0,1,2,3]
print('list1:', list1)
list2 = [[0] ,[1] ,[2] ,[3]]
print('list2:',list2)
list3 = [[0 ,1] ,[2 ,3]]
print('list3:',list3)
array1 = np.array ([0,1,2,3])
print('array1:',array1)
array2 = np.array ([[0] ,[1] ,[2] ,[3]])
print('array2:',array2)
array3 = np.array ([[0 ,1] ,[2 ,3]])
print('array3:',array3)

print(np.pi)

#this is a comment

#printing bigger arrays
print(np.arange(4),'\n',
      np.arange(0,2,0.5),'\n',
      np.linspace(0,1.5,4))

#indexing lists and arrays
list1 = [1,2,3,4,5]
array1 = np.array(list1)    #defines a list as an array
print('list1 :',list1[0],list1[4])
print('array1:',array1[0],array1[4])
array2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
list2 = list(array2)        #makes a list out of an array
print('array2:',array2[0,2],array2[1,4])
print('list2 :',list2[0],list2[1])

#use a colon to print an entire row or coloumn
print(array2[:,2], array2[0,:])

#defining a matrix of 1's or 0's
print('1x3:',np.zeros(3))
print('2x2:',np.zeros((2,2)))
print('2x3:',np.ones((2,3)))

#plotting graphs in python
#define variables
steps = 0.1     # defining step size
x = np.arange(-2,2+steps,steps) #final value is 2 + steps to include the value '2'

y1 = x+2
y2 = x**2

#plot code
plt.figure(figsize=(12,8))  #creates a new figure with a custom figure size

plt.subplot(3,1,1)  #subplot 1: format(row, column, number)
plt.plot(x,y1)  # choose varaibles for plot
plt.title('Sample Plots for Lab 1') #title for whole figure
plt.ylabel('Subplot 1')
plt.grid(True)

plt.subplot(3,1,2)  #subplot 1: format(row, column, number)
plt.plot(x,y2)  # choose varaibles for plot
plt.ylabel('Subplot 2')
plt.grid(which='both')

plt.subplot(3,1,3)  #subplot 1: format(row, column, number)
plt.plot(x,y1,'--r',label='y1')
plt.plot(x,y2,'o',label='y2')
plt.axis([-2.5,2.5,-0.5,4.5])   #define axis
plt.grid(True)
plt.legend(loc='lower right') #prints legend
plt.xlabel('x')
plt.ylabel('Subplot 3')
plt.show()  #must be included for plots to be visible

#complex numbers
cRect = 2 + 3j
print(cRect)

cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
print(cPol) #stored in rectangular form

cRect2 = np.real(cPol) + 1j*np.imag(cPol)
print(cRect2)

#to get imaginary numbers to come out of a function, add a 0j at some point
print(np.sqrt(3*5 - 5*5 + 0j))



