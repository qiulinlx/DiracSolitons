import matplotlib.pyplot as plt
import numpy as np
from sympy import cot
import cmath, math
from scipy.fft import fft, ifft
import pandas as pd
from matplotlib import cm

alpha=np.pi/3
w=np.cos(2*alpha)
k=np.sqrt(1-w**2)

def phi(x):
    return k/np.cosh(k*x+alpha*(0+1j))

def sigma(x):
    return -k/np.cosh(k*x-alpha*(0+1j))

def u(x,t):
    time= np.exp(-(0+1j)*w*t)
    return time *sigma(x)

N=100
h=2*np.pi/N
t=np.linspace(0,50,500)
#print(t)

x=np.arange(-np.pi, np.pi, h)
X,T=np.meshgrid(x,t)
#print(X,T)
Z=u(X,T)
#print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X,T, Z.real, cmap=cm.jet)
plt.show()