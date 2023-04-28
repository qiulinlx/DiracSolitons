import matplotlib.pyplot as plt
import numpy as np
import cmath, math
from scipy.fft import fft, ifft
import pandas as pd

alpha=np.pi/4
w=np.cos(2*alpha)
k=np.sqrt(1-w**2)

def phi(x):
    return k/np.cosh(k*x+alpha*(0+1j))

def sigma(x):
    return -k/np.cosh(k*x-alpha*(0+1j))

N=100
x=[]

for j in range(N+1):
    xi=np.cos(j*np.pi/(N))
    xi=xi
    x.append(xi)
    
x=np.array(x) 
#print(x)
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

print(Cheb(N,x))

v=phi(x)
w=np.dot(Cheb(N,x),v)
print(w)
#w=w[5:-5]
#x=x[5:-5]
plt.plot(x,w.real)
plt.xlabel('Postion x')
plt.ylabel("du/dx")
plt.title('Derivative of u(x,t)')
plt.show()   

def anadiff(x):
    return -k**2*(1/np.cosh(k*x+alpha*(0+1j)))*np.tanh(k*x+alpha*(0+1j)) 

sol=anadiff(x)

#print(sol)

acc= (sol-w.real)
plt.plot(x,acc)
plt.xlabel('Postion x')
plt.ylabel("Error")
plt.title('Accuracy of Approximation')
plt.show()