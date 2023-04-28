import matplotlib.pyplot as plt
import numpy as np
from sympy import cot
import cmath, math
import pandas as pd
from matplotlib import cm
from scipy.linalg import toeplitz
from scipy.fft import fft, ifft

import time
start=time.time()

alpha=np.pi/9

w=np.cos(2*alpha)
K=np.sqrt(1-w**2)

def phi(x):
    return K/np.cosh(K*x+alpha*(0+1j))

def sigma(x):
    return -K/np.cosh(K*x-alpha*(0+1j))

def dSn(i):
    return 0.5*(-1)**i*cot(h*i/2)

N=100
L=5

h= 2*np.pi/N

x=np.linspace(-L/np.pi,L/np.pi, N)

def u(x,t):
    time=np.exp(-w*t*(0+1j))
    return time*phi(x)


def v(x,t):
    time=np.exp(-w*t*(0+1j))
    return time*sigma(x)

#Times
times=[]
tau=0.01
tstep=400

def Diff(x):
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
    diff=diff/(L*np.pi)
    return diff


#initial conditions

u=u(x,0)
v=v(x,0)

du=np.dot(Diff(x),u)
dv=np.dot(Diff(x),v)

bu= u-tau*(du-0j*(v+np.abs(v)**2*u))
bv= v-tau*(du-0j*(u+np.abs(u)**2*v))

y1 = np.zeros((tstep,N),dtype="complex_")
y2 = np.zeros((tstep,N),dtype="complex_")

for n in range (tstep):
    t=n*tau
    times.append(t)
    du=np.dot(Diff(x),u)
    dv=np.dot(Diff(x),v)

    nu=2*tau*(-1j*(v+np.abs(v)**2*u)+du)+bu
    nv=2*tau*(-1j*(u+np.abs(u)**2*v)-dv)+bv

    bu=u
    u=nu

    bv=v
    v=nv

    #plt.plot(x,du)
   # plt.show()

    y1[n]=u
    y2[n]=v
    #plt.plot(du)
    #plt.show()


X,T = np.meshgrid(x,times)
end=time.time()

#print(np.shape(X),np.shape(u),np.shape(v))

fig = plt.figure()
ax = plt.axes(projection='3d') #We've created the figure!

ax.plot_surface(X,T, y1.real, cmap=cm.jet, label= 'u')
# Set labels and zticks
ax.set(xlabel='x',ylabel='t')
#fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Height')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d') #We've created the figure!

#ax.plot_surface(X,T, y1.real, cmap=cm.jet)
ax.plot_surface(X,T, y2.real, cmap=cm.jet, label= 'v')

#C=ax.plot_surface(T,X,np.sqrt(y1.real**2+y2.real**2), cmap=cm.jet)
# Set labels and zticks
ax.set(xlabel='x',ylabel='t')
#fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Height')
plt.show() 


tt=end-start 
print(end-start)