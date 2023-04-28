import matplotlib.pyplot as plt
import numpy as np
from sympy import cot
import cmath, math
from scipy.fft import fft, ifft
import pandas as pd

i=complex(0,1)
t=90

alpha=np.pi/3
w=np.cos(2*alpha)
k=np.sqrt(1-w**2)

def phi(x):
    return k/np.cosh(k*x+alpha*i)

def sigma(x):
    return -k/np.cosh(k*x-alpha*i)

N=100
h=2*np.pi/N
x=np.arange(-np.pi, np.pi, h)
vx=x
ux=x
#plt.plot(x, np.exp(i*w*t)*phi(x), label='u(x,t)')
#plt.plot(vx, sigma(x), label='psi(x)', color= 'green')
plt.plot(ux, phi(x), label='phi(x)', color= 'blue')

plt.xlabel('Postion x')
plt.title('Initial form of Solitons')
plt.legend()
plt.show()