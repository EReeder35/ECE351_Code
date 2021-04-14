# -*- coding: utf-8 -*-
############################################
#                                          #
#   Ethan Reeder                           #
#   ECE351-53                              #
#   Lab #11                                #
#   Due 20April2021                        #
#                                          #
############################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Zplane function

#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.
#
#
#
# Modified by Drew Owens in Fall 2018 for use in the University of Idaho's 
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# (ECE 351)
#
# Modified by Morteza Soltani in Spring 2019 for use in the ECE 351 of the U of
# I.
#
# Modified by Phillip Hagen in Fall 2019 for use in the University of Idaho's  
# Department of Electrical and Computer Engineering Signals and Systems I Lab 
# (ECE 351)

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function."""

    #import numpy as np
    #import matplotlib.pyplot as plt
    from matplotlib import patches    
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

#%%

# Part 1 Task 3

num = [2, -40]
den = [1, -10, 16]

res = sig.residuez(num, den)

print('Part 1 Task 3 R, P, and K Values')
print('R: ', res[0])
print('P: ', res[1])
print('K: ', res[2])

#%%

# Part 1 Task 4

z, p, k = zplane(num, den)

#%%

# Part 1 Task 5
w, h = sig.freqz(num, den, whole = True)
mag1 = np.abs(h)
mag1 = 20* np.log10(mag1)

w = w / np.pi

plt.figure(figsize = (12,8))
plt.title('scipy.signal.bode Function')
plt.subplot(2,1,1)
plt.ylabel('|H| (dB)')
plt.plot(w, mag1)
plt.grid()

plt.subplot(2,1,2)
plt.ylabel('/_H (degrees)')
plt.plot(w, np.angle(h))
plt.grid()

plt.show()
