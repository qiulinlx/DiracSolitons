import matplotlib.pyplot as plt
import numpy as np
from sympy import cot
import cmath, math
import pandas as pd
from scipy.linalg import toeplitz


#i=complex(0,1)

alpha=np.pi/3
w=np.cos(2*alpha)
k=np.sqrt(1-w**2)

def phi(x):
    return k/np.cosh(k*x+alpha*(0+1j))

def sigma(x):
    return -k/np.cosh(k*x-alpha*(0+1j))

#Fourier diff matrix
#Interval x e [-L,L]
L=10
N=20
h=2*np.pi/N
x=np.linspace(-L/np.pi, L/np.pi, N)
print(len(x))

def dSn(i):
    return 0.5*(-1)**i*cot(h*i/2)

x=np.array(x)
v=phi(x)
diff=np.empty([N,N])

col=[]
col.append(0)

for n in range(1,N):
    c= dSn(n)
    col.append(c)
col=np.array(col)
#print(col)
row=-col
#print (row)

diff =toeplitz(col, row)
diff=np.array(diff, 'complex')
diffl=diff/(L*np.pi)
#print(type(diff[0,1]))

w=np.dot(diff, v)
#w=list(w)
#print(type(w))
#print(w.real)


#w=np.round(w,5)

print(w.real)
plt.plot(x, w.real)
plt.xlabel('Postion x')
plt.ylabel("du/dx")
plt.title('Derivative of u(x,t)')
#plt.plot(x, v)
#plt.show()


#actual 

def anadiff(x):
    return (-k**2*(1/np.cosh(k*x+alpha*(0+1j)))*np.tanh(k*x+alpha*(0+1j))) 

sol=anadiff(x)
#sol=np.round(sol, 5)

#print(sol)

acc= (w.real-sol.real)
#plt.plot(x,sol)
plt.show() 
plt.plot(x,np.abs(acc))
plt.xlabel('Postion x')
plt.ylabel("Error")
plt.title('Accuracy of Approximation')
plt.show()