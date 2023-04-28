import matplotlib.pyplot as plt
import numpy as np
from sympy import cot
import cmath, math
from scipy.fft import fft, ifft
import pandas as pd
from matplotlib import cm

alpha=(np.pi)/4-0.15
w=np.cos(2*alpha)
k=np.sqrt(1-w**2)
L=5
def phi(x):
    return k/np.cosh(k*x+alpha*(0+1j))

def sigma(x):
    return -k/np.cosh(k*x-alpha*(0+1j))

N=100
x=[]

for j in range(N+1):
    xi=np.cos(j*np.pi/(N))
    xi=L*np.arctanh(xi)
    x.append(xi)

x=x[1:-1]
x.append(N)
x.insert(0,N)
x=np.array(x)

z=(1/L)*(np.cosh(x/L))**(-2) 
#print(v)
diag=np.diag(z)
    
def Cheb(N,x):
    diff=np.empty([N+1,N+1])

    diff[0,0]= (2*(N)**2+1)/6 #y
    diff[0, N]=(-1)**N*(0.5) #y
    diff[N,0]= -0.5*(-1)**(N)  #y
    diff[N, N]= -(2*(N)**2+1)/6 #y

    for i in range (1,N):
        diff[0,i]= (2)*((-1)**i)/(1-x[i]) #top
        diff[N,i]= (-2*(-1)**(N+i))/(1+x[i]) #bottom

        diff[i,0]= (-1/2)*((-1)**i)/(1-x[i]) #column 0
        diff[i,N]= (0.5*(-1)**(N+i))/(x[i]+1)

        for j in range(1,N):

            if i!=j:
                diff[i,j]= ((-1)**(i+j))/(x[i]-x[j]) #Inner MESS
            if i==j:
                diff[i,j]= (-x[j])/(2*(1-(x[j])**2)) #Diagonal
    return diff

Dzz=Cheb(N, x)
#print(Dzz)

#print(Dzzz)

Dz=np.dot(diag,Dzz)
#print(Dz)
u=phi(x)
v=sigma(x)

a=u.real
b=u.imag
f=v.real
g=v.imag

m11=Dz
#print(m11)

ma=(-w-f**2-g**2)
m12=np.diag(ma)

mb=-2*b*f
m13=np.diag(mb)

mc=(-1-2*b*g)
m14=np.diag(mc)

q=np.concatenate((m11,m12,m13, m14),axis=1)

md=(w+f**2+g**2)
m21=np.diag(md)

m22=Dz

me=(1+2*a*f)
m23=np.diag(me)

mf=2*a*g
m24=np.diag(mf)

r=np.concatenate((m21,m22,m23, m24),axis=1)

mg=-2*a*g
m31=np.diag(mg)

mh=-1-2*b*g
m32=np.diag(mh)

m33=-Dz

mi=-w-a**2-b**2
m34=np.diag(mi)

s=np.concatenate((m31,m32,m33, m34),axis=1)

mj=1+2*a*f
m41=np.diag(mj)

mk= 2*b*f
m42=np.diag(mk)

ml=w+a**2+b**2
m43=np.diag(ml)

m44=-Dz

t=np.concatenate((m41,m42,m43, m44),axis=1)

Diff=np.concatenate((q,r,s,t),axis=0)
#print(Diff)

e, v=np.linalg.eig(Diff)

e=np.array(e)

plt.plot(e.real, e.imag, '.', markersize=5)
plt.title('Eigenvalues of Linearised System with Alpha=pi/10')
plt.xlabel("Real)")
plt.ylabel('Imag')
#plt.show()


order=[]
for i in range(len(e)):
    if e.imag[i] != 0:
        order.append(e.real[i])

ordered=np.sort(order)

print(ordered[-1])