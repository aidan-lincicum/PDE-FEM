from numpy import *
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d
from numpy.linalg import norm

#Create region
fig,ax = plt.subplots()
y0 = -2+sqrt(3.85)
x0 = sqrt(4*y0+2.4)
x = linspace(-x0,x0,300)
ax.plot(x,-0.6+0.25*x**2)
ax.plot(x,-sqrt(1.5**2-x**2))
ax.plot([x0,-x0],[y0,y0],'ro')
ax.set_aspect("equal")

# upper edge
x1 = linspace(-x0,x0,20)
y1 = -0.6+0.25*x1**2 

# lower edge
th = linspace(pi,2*pi,23)
x2 = 1.5*cos(th)
y2 = 1.5*sin(th)
x2 = x2[y2<y0]
y2 = y2[y2<y0]

# grid in interior
xx,yy = meshgrid(linspace(-1.5,1.5,21),linspace(-2.05,0.95,21))
xx[::2,:] += 0.5*3/20
keep = logical_and(xx**2+yy**2<1.43**2, yy<-0.66+0.25*xx**2)

x3 = xx[keep].reshape(-1)
y3 = yy[keep].reshape(-1)

xtri, ytri = hstack((x1,x2,x3)),hstack((y1,y2,y3))

de = Delaunay(stack((xtri,ytri)).T)
triangles = de.points[de.simplices,:]
centroids = triangles.sum(axis=1)/3
keep = centroids[:,1] < -0.6+0.25*centroids[:,0]**2
de.simplices = de.simplices[keep,:]
triangles = de.points[de.simplices,:]
fig, ax = plt.subplots()
delaunay_plot_2d(de,ax=ax)
ax.set_aspect("equal")

#Find gradient of the triangles
def mygrad(a,b,c):
    p = c-b
    q = a-b
    alpha = (q*p).sum(axis=-1)/(p*p).sum(axis=-1)
    r = q - alpha[:,None]*p
    rn = norm(r,axis=-1)
    gradphi = r/rn[:,None]**2
    area = 0.5*rn*norm(p,axis=-1)
    return gradphi,area 

grads1,areas1 = mygrad(triangles[:,0],triangles[:,1],triangles[:,2])
grads2,areas2 = mygrad(triangles[:,1],triangles[:,2],triangles[:,0])
grads3,areas3 = mygrad(triangles[:,2],triangles[:,0],triangles[:,1])
j1,j2,j3 = de.simplices[:,0], de.simplices[:,1],de.simplices[:,2]

# from scipy.sparse import coo_matrix
# A = coo_matrix(((grads1*grads1).sum(axis=-1),(j1,j1)),shape=())