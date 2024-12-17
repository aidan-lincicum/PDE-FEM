import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.integrate as integrate

def f(x):
    return(math.cos(x))

def real(x):
    return(1/(np.cos(3)*3)*(np.sin(3*x) + 6*np.cos(3-3*x)))

def first_order(start, end, pts):
    h = (end-start)/pts

    #ODE of form au'+ bu = f(x)
    a = 4
    b = 3

    #u(0) = ic
    ic = 1

    K = np.zeros((pts,pts))
    K[0,0] = (2*b*h)/3
    K[0,1] = (b*h + 3*a)/6

    K[pts-1,pts-1] = (3*a + 2*b*h)/6
    K[pts-1,pts-2] = (b*h-3*a)/6

    ln = np.array([(-3*a + b*h)/6, 2*b*h/3, (3*a + b*h)/6])
    for i in range(1,pts-1):
        K[i,i-1:(i+2)] = ln

    b1 = np.zeros(pts)
    b1[pts-1] = (integrate.quad(lambda x: (x - (pts-1)*h)*f(x),h*(pts-1),h*pts)[0])/h

    for i in range(0,pts):
        b1[i] = (integrate.quad(lambda x: (x-h*i)*f(x),h*i,h*(i+1))[0] + integrate.quad(lambda x: (h*(i+2)-x)*f(x), h*(i+1),h*(i+2))[0])/h
    b1[0] = b1[0] - ((-3*a+b*h)/6)*ic

    q = np.linalg.solve(K,b1)
    q = np.insert(q,0,ic)
    return(q)


def second_order(start, end, pts):
    h = (end-start)/pts

    #ODE of form au''+ bu' + cu = f(x)
    a = 1
    b = 0
    c = 9

    #u(0) = ic1
    ic1 = 2
    #u'(end) = ic2
    ic2 = 1

    K = np.zeros((pts, pts))
    K[0,0] = a/h + 2*c*h/3 - a*3/h
    K[0,1] = -a*2/h + b/2 * c*h/3 + a*3/h

    K[pts-1,pts-2] = -a*(pts-1)/h - b/2 + c*h/6 + a*pts/h
    K[pts-1,pts-1] = a*(pts-1)/h + b/2 + c*h/3 - a*pts/h

    for i in range(1,pts-1):
        K[i,i-1] = -a*(i)/h - b/2 + c*h/6 + a*(i+1)/h
        K[i,i] = a*i/h + 2*c*h/3 - a*(i+2)/h
        K[i,i+1] = -a*(i+1)/h + b/2 + c*h/6 + a*(i+2)/h
    
    b1 = np.zeros(pts)
    for i in range(1,pts):
        first_int = integrate.quad(lambda x: (x - (i+1)*h)*f(x), i*h, (i+1)*h)[0]
        second_int = integrate.quad(lambda x:((i+2)*h - x)*f(x), (i+1)*h, (i+2)*h)[0]
        b1[i-1] = (first_int + second_int)/h
    b1[pts-1] = (integrate.quad(lambda x: (x-(pts-1)*h)*f(x),(pts-1*h),pts*h)[0])/h - a*ic2   
    b1[0] -= (-a/h - b/2 + c*h/6 + a*2/h)*ic1

    q = np.linalg.solve(K,b1)
    q = np.insert(q,0,ic1)
    return(q)

q = first_order(0,10,50)
x = np.linspace(0,10,50+1)
xreal = np.linspace(0,10,1000)
y = real(xreal)

# q = second_order(0,10,1000)
# x = np.linspace(0,10,1000+1)

fig,ax = plt.subplots()
ax.plot(x,q)
ax.grid()
plt.show()

error = np.zeros(101)

for i in range(2,103):
    q = second_order(0,1,i)
    x = np.linspace(0,1,i+1)
    for j in range(0,len(q)):
        error[i-2] += (q[j] - real(x[j]))**2
    error[i-2] = error[i-2]**0.5

print(error)

fig,ax = plt.subplots()
ax.semilogy(range(2,103),error)
ax.grid()
plt.show()