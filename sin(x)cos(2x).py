#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
from numpy.fft import fft, fftfreq, ifft

N = 1000
loops = 1

### sin(x)cos(2x) ###
x = np.linspace(0,  2 * loops *np.pi, loops * N)
y = np.sin(x) * np.cos(2*x)

############# Plotting Original Signal ##############
plt.title('Original Signal: sin(x)cos(2x)')
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.plot(x, y, label = "Signal")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.show()

########### Fourier Transform ##############
#Normal
freq =  (N/2) * np.arange(-1.0, 1.0 , 2.0/N)
fft_val = (2.0/N) * abs(fft(y))

#Raw 
fft_raw = [0] * N
for i in range (int(N-1), int(N/2 - 1) , -1):
    fft_raw[int(i - N/2)] = fft_val[i]
for i in range (0,int(N/2)):
    fft_raw[int(N/2 + i)] = fft_val[i]

#Real
freq_real = (N/2) * np.arange(0, 1, 2.0/N)
fft_real = fft_val[0 : int(N/2)]

#Inverse
ifft_val = ifft(fft(y))

################ Plotting FT ##################
plt.title('Fourier Transform')
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
#plt.plot(freq_real, fft_real, label = "Real FT of Signal")
plt.plot(freq, fft_raw, label = "Raw FT of Signal")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.xlim(-20, 20)
plt.show()

#### Plotting IFFT ####
plt.title('Inverse Fourier Transform')
plt.ylabel("Intensity")
plt.xlabel("x")
plt.plot(x, ifft_val, label = "IFFT of Signal")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#plt.xlim(-1, 10)
plt.show()

#### Error of fft to ifft
ifftError =  100* abs( (y - ifft_val) / y)

plt.plot(x[1:N-1], ifftError[1:N-1], label = "Error")
plt.title('x vs. Error of Function and Inverse FT')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.show()