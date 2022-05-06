'''
Test filters to smooth lumination results
'''

from scipy import signal
import numpy as np
from readVideo import loadRes
from matplotlib import pyplot as plt
from LMSfilter import *

def mean53(data):
    n = len(data)
    if n<5:
        print('err: len data < 5')
        return
    y = np.zeros((n,1))
    y[0] = (69*data[0] + 4*(data[1] + data[3]) - 6*data[2] - data[4]) /70
    y[1] = (2*(data[0] + data[4]) +27*data[1] + 12*data[2] - 8*data[3]) /35
    for j in range(2,n-2):
        y[j] = (-3*(data[j-2] + data[j+2]) + 12*(data[j-1] + data[j+1]) + 17*data[j]) /35
    y[n-2] = (2*(data[n-1] + data[n-5]) + 27*data[n-2] + 12*data[n-3] - 8*data[n-4]) /35
    y[n-1] = (69*data[n-1] + 4*(data[n-2] + data[n-4]) - 6*data[n-3] -data[n-5]) /70
    return y

def dft(xn,dT, ifplot=False):
    # TEST
    # wn = 325
    # dT = 1/100/3 # ws>3*wn
    # t = dT* np.array(range(1000))
    # xn = np.sin(2*np.pi*wn*t)
    
    # fft
    n = len(xn)
    x=1/n*np.fft.fft(xn,np.size(xn,0),axis=0) # half amplitude 0.5 for sin(t)
    freq=np.fft.fftfreq(np.size(xn,0),dT)
    x = np.fft.fftshift(x)
    freq = np.fft.fftshift(freq)

    if ifplot:
        plt.ion()
        plt.figure(2)
        # amplitude spectrum
        plt.plot(freq,abs(x))
        plt.ioff()
        plt.show()

    return freq,x

def minInN(xn,nN = 10):
    n = len(xn)
    if np.mod(n,nN)==0:
        n2 = n//nN
    else:
        n2 = n//nN + 1
    x2 = np.zeros((n2,1))
    j = 0
    for i in range(0,n,nN):
        if i+nN <= n:
            x2[j] = np.mean(xn[i:i+nN])
        else:
            x2[j] = np.mean(xn[i:n])
        j = j +1
    return x2


if __name__=='__main__':
    data = loadRes('Res.xlsx','2018.07.07.1115 -7冷启动')

    i = 4
    dT = 0.5

    freq = 1/dT # [Hz] sample rate
    y = data[:,i]


    # amplitude spectrum
    # dft(y,dT,True)
    
    nTest = 5
    plt.ion()

    fig,axs = plt.subplots(nTest,1)
    axs[0].plot(y) 
    axs[0].set_ylabel('origin')

    axs[1].plot(mean53(y)) 
    axs[1].set_ylabel('mean53')

    ## wave filtering
    Wn = 2*dT*0.02 #  Wn=2*截止频率/采样频率
    b, a = signal.butter(N=6,Wn=2*Wn/freq,btype='lowpass')
    filtedData = signal.filtfilt(b,a,y)
    axs[2].plot(filtedData)
    axs[2].set_ylabel('lowpass: shutdown freq. 0.02')

    yn,en = LMS(y,None,64,1e-4,len(y))
    axs[3].plot(yn)
    axs[3].set_ylabel('LMS')
    # plt.plot(y-filtedData)

    axs[4].plot(minInN(y,5))
    axs[4].set_ylabel('min in N')

    plt.ioff()
    plt.show()

    1