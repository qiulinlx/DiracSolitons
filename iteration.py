import numpy as np
import matplotlib.pyplot as plt
import cmath, math
alpha=np.pi/3
w=np.cos(2*alpha)
K=np.sqrt(1-w**2)

def f(x,y1, y2):
    return -1j*(w*y1+y2+np.abs(y2)**2*y1)

def g(x,y1, y2):
    return 1j*(w*y2+y1+np.abs(y1)**2*y2)

N=100
L=10

h= 0.01


x=np.linspace(-L/np.pi,L/np.pi, N)
y1=0.01
#print(y1)
y2=-0.01

#print(f(x,y1,y2))
phi=[]
phi.append(y1)
psi=[]
psi.append(y2)
xl=[]
x=-L/np.pi
xl.append(x)
for i in range(900):

    k1=h*f(x,y1,y2)
    k2=h*f(x+h,y1+(1/2)*k1,y2+(1/2)*k1)
    
    y1n=y1+(1/2)*(k1+k2)
    k12=h*g(x,y1,y2)
    k22=h*g(x+h,y1+(1/2)*k1,y2+(1/2)*k1)
    
    y2n=y2+(1/2)*(k12+k22)

    y2=y2n
    y1=y1n 
    x+=h
    xl.append(x)
    phi.append(y1)
    psi.append(y2)


print(phi)

plt.plot(xl,psi, color='green')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Iterative solution Psi using RK2')

plt.show()