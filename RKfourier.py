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
alpha=np.pi/3
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

#initial conditions

u=u(x,0)
bu=u
v=v(x,0)
bv=v
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

y1 = np.zeros((tstep,N),dtype="complex_")
y2 = np.zeros((tstep,N),dtype="complex_")

#Runge-Kutta
def f(u,du,v,t):
    return -1j*(v+np.abs(v)**2*u)+du

def g(v,dv, u,t):
    return -1j*(u+np.abs(u)**2*v)-dv

y1[0]=u
y2[0]=v

for i in range (tstep):
    t=i*tau
    times.append(t)

    du=np.dot(Diff(x),u)
    dv=np.dot(Diff(x),v)

    k1=tau*f(u,du,v, t)
    k2=tau*f(u+(tau/2)*k1, du+(tau/2)*k1, v+(tau/2)*k1, t+(tau/2))
    k3=tau*f(u+(tau/2)*k2, du+(tau/2)*k2, v+(tau/2)*k2, t+(tau/2))
    k4= tau*f(u+tau*k3, du+tau*k3, v+tau*k3, t+tau)
    nu= u+(1/6)*(k1+2*k2+2*k3+k4)

    vk1=tau*g(v,dv,u, t)
    vk2=tau*g(v+(tau/2)*vk1, dv+(tau/2)*vk1, u+(tau/2)*vk1, t+(tau/2))
    vk3=tau*g(v+(tau/2)*vk2, dv+(tau/2)*vk2, u+(tau/2)*vk2, t+(tau/2))
    vk4= tau*g(v+tau*vk3, dv+tau*vk3, u+tau*vk3, t+tau)
    nv= v+(1/6)*(vk1+2*vk2+2*vk3+vk4)

    bu=u
    bv=v
    u=nu
    v=nv

    y1[i]=u
    y2[i]=v
    #plt.plot(x,u.real)
    #plt.plot(x,v.real)
    #plt.show()
end=time.time()

X,T = np.meshgrid(x,times)

#print(np.shape(X),np.shape(u),np.shape(v))

fig = plt.figure()
ax = plt.axes(projection='3d') #We've created the figure!

ax.plot_surface(X,T, y1.real, cmap=cm.jet)
#ax.plot_surface(X,T, y2.real, cmap=cm.jet)


#C=ax.plot_surface(T,X,np.sqrt(y1.real**2+y2.real**2), cmap=cm.jet)
# Set labels and zticks
ax.set(xlabel='x',ylabel='t')
#fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Height')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d') #We've created the figure!

ax.plot_surface(X,T, y2.real, cmap=cm.jet)


#C=ax.plot_surface(T,X,np.sqrt(y1.real**2+y2.real**2), cmap=cm.jet)
# Set labels and zticks
ax.set(xlabel='x',ylabel='t')
#fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Height')
plt.show()




tt=end-start 
print(end-start)