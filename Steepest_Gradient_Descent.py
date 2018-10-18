import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m 

#  Reading the different text file   
filename = ['quadratic.opt','quartic.opt','quad_unknown.opt',
    'quartic_saddle.opt','quartic_saddle_unknown.opt','quartic_unknown.opt']

file = open(filename[0],'r').readlines()

#for lines in file: 
#    print(lines)

N = int(file[0])

f = lambda x: eval(file[1])

def g(x,N):
    
    doc = [txt.strip('\n') for txt in file]     #purpose of doc is so that file[2]=='unknown' then python actually understands it
    if doc[2] == 'unknown':
        x_k = x
        e = 1e-6
        Grad = []
        for i in range(2):
            x_k[i] = x_k[i]+e                   # I am trying to achieve df/dx_i ~ f(x+e)-f(x-e)/2e
            h = f(x_k)                          # But I did not know how to set a global variable so had to do it a long way
            x_k[i] = x_k[i]-2*e
            d = f(x_k)
            df = (h-d)/(2*e)
            Grad.append(df)
            x_k[i] = x_k[i]+e
        return np.array(Grad)   
    else:
        list = []
        for t in file[2].split():
            list.append(eval(t))
        return np.array(list)        

x_0 = [float(t) for t in file[3].split()]
TOL = float(file[4])
maxit = int(file[5])

def grad_descent(f, g, N,x_0, maxit, TOL, verbose=False):
    '''This function calculates the point where f(x) has a minimum'''
    
    msg = "Maximum number of iterations reached."
    x = x_0
    g_old = g(x,N)
    L = np.linspace(0,1,100)
    s_k = min([f(x-s*g_old) for s in L])     #This is me trying to do line search such that s_k = min_s f(x_k-sg(x_k))
    print('Starting s_k for line search = %0.4f' %(s_k))
    W = [x]
    for cont in range(maxit):
        if verbose:
            print("k: {}, x: {},f: {}, g: {}".format(cont, x,np.round(f(x),6), np.round(np.linalg.norm(g_old,2),6)))
        dx = -s_k*g_old
        x = x + dx
        W.append(x)
        grad = g(x,N)                 #Here I am using Barzilai and Borwein method to calculate s_k
        dg = grad - g_old          # s_k = dx^Tdg/||dg||^2 
        g_old = g(x,N)
        s_k = dx.dot(dg)/dg.dot(dg)
        if np.linalg.norm(g(x,N),2) < TOL:
            msg = "Extremum found with desired accuracy with %d iterations" %(cont)
            break
    return x, f(x), msg,W

grad_descent(f, g, N,x_0, maxit, TOL, verbose=True)[3]
x_p = grad_descent(f, g, N,x_0, maxit, TOL, verbose=False)[0]     #trying to extract x* which gives the minima

'''This calculation below is to plot the contourplots '''

x_list = np.linspace(x_p[0]-x_0[0],x_0[0]+x_p[0],100)
y_list = np.linspace(x_p[1]-x_0[1],x_0[1]+x_p[1],100)
#x_list = np.linspace(-2,2,100)
#y_list = np.linspace(-2,2,100) 

#W = np.array(grad_descent(f, g, N,x_0, maxit, TOL, verbose=False)[3])
#X_1 = [W[i,0] for i in range(len(W))]
#Y_1 = [W[i,1] for i in range(len(W))]

X,Y = np.meshgrid(x_list,y_list)

#Z = Y**4+X**4+2*X*Y-3*X**2
Z = f([X,Y])

plt.contour(X,Y,Z)

plt.title('Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.gcf().set_size_inches(10.5, 8)
plt.show()
#plt.plot(X_1,Y_1)
#plt.show()






