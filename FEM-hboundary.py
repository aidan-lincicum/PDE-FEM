from numpy import *
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay, delaunay_plot_2d
from numpy.linalg import norm, inv
from scipy.sparse.linalg import spsolve
from PIL import Image, ImageOps

#Constants
num_points = 268
xmin = -1.5
xmax = 1.5
ymin = -1.5
ymax = 0


#Create region
fig,ax = plt.subplots()
x1 = linspace(-3,-0.5,25)
y1 = full(25, 2)
x2 = full(17, -0.5)
y2 = linspace(2, 0.25, 17)
x3 = linspace(-0.5, 0.5, 10)
y3 = full(10, 0.25)
x4 = full(17, 0.5)
y4 = linspace(0.25,2, 17)
x5 = linspace(0.5, 3, 25)
y5 = full(25, 2)
x6 = full(40, 3)
y6 = linspace(2,-2,40)
x7 = linspace(3,0.5, 25)
y7 = full(25, -2)
x8 = full(17, 0.5)
y8 = linspace(-2,-0.25, 17)
x9 = linspace(0.5,-0.5,10)
y9 = full(10, -0.25)
x10 = full(17, -0.5)
y10 = linspace(-0.25,-2,17)
x11 = linspace(-0.5,-3, 25)
y11 = full(25, -2)
x12 = full(40, -3)
y12 = linspace(-2,2,40)

# grid in interior
xx,yy = meshgrid(linspace(-2.95,2.95,60),linspace(-1.95,1.95,40))

keep = logical_or(logical_or(logical_and(logical_and(xx>-3,xx<-0.5), logical_and(yy>-2,yy<2)), \
                             logical_and(logical_and(xx>0.5,xx<3), logical_and(yy>-2,yy<2))), \
                             logical_and(logical_and(xx>-0.5,xx<0.5), logical_and(yy>-0.25,yy<0.25)))

x13 = xx[keep].reshape(-1)
y13 = yy[keep].reshape(-1)

xtri, ytri = hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13)),hstack((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13))

de = Delaunay(stack((xtri,ytri)).T)
neighbors = de.neighbors
triangles = de.points[de.simplices,:]
centroids = triangles.sum(axis=1)/3
keep = logical_or(logical_or(logical_and(logical_and(centroids[:,0]>=-3,centroids[:,0]<=-0.5), logical_and(centroids[:,1]>=-2,centroids[:,1]<=2)), \
                             logical_and(logical_and(centroids[:,0]>=0.5,centroids[:,0]<=3), logical_and(centroids[:,1]>=-2,centroids[:,1]<=2))), \
                             logical_and(logical_and(centroids[:,0]>=-0.5,centroids[:,0]<=0.5), logical_and(centroids[:,1]>=-0.25,centroids[:,1]<=0.25)))
de.simplices = de.simplices[keep,:]
triangles = de.points[de.simplices,:]

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
j1,j2,j3 = de.simplices[:,0], de.simplices[:,1], de.simplices[:,2]

pts = len(de.points)
A1 = coo_matrix(((grads1 * grads1).sum(axis=-1)*areas1,(j1,j1)),shape=(pts,pts))
A2 = coo_matrix(((grads1 * grads2).sum(axis=-1)*areas1,(j1,j2)),shape=(pts,pts))
A3 = coo_matrix(((grads1 * grads3).sum(axis=-1)*areas1,(j1,j3)),shape=(pts,pts))
A4 = coo_matrix(((grads2 * grads1).sum(axis=-1)*areas1,(j2,j1)),shape=(pts,pts))
A5 = coo_matrix(((grads2 * grads2).sum(axis=-1)*areas1,(j2,j2)),shape=(pts,pts))
A6 = coo_matrix(((grads2 * grads3).sum(axis=-1)*areas1,(j2,j3)),shape=(pts,pts))
A7 = coo_matrix(((grads3 * grads1).sum(axis=-1)*areas1,(j3,j1)),shape=(pts,pts))
A8 = coo_matrix(((grads3 * grads2).sum(axis=-1)*areas1,(j3,j2)),shape=(pts,pts))
A9 = coo_matrix(((grads3 * grads3).sum(axis=-1)*areas1,(j3,j3)),shape=(pts,pts))
K = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9
Knew = K.toarray()[int(num_points):,int(num_points):]

M = zeros((pts, pts))
for i in range(0, len(j1)):
    M[j1[i], j1[i]] += areas1[i] / 6
    M[j1[i], j2[i]] += areas1[i] / 12
    M[j1[i], j3[i]] += areas1[i] / 12
    M[j2[i], j1[i]] += areas1[i] / 12
    M[j2[i], j2[i]] += areas1[i] / 6
    M[j2[i], j3[i]] += areas1[i] / 12
    M[j3[i], j1[i]] += areas1[i] / 12
    M[j3[i], j2[i]] += areas1[i] / 12
    M[j3[i], j3[i]] += areas1[i] / 6

Mnew = M[int(num_points):, int(num_points):]
Minv = inv(Mnew)

im1 = Image.open("imgs/cat.jpg")
im2 = ImageOps.grayscale(im1)

numpydata = asarray(im2)
def icfunc(x, y):
    return 1 if x < -0.5 else 0

ic = zeros(pts)
for i in range(pts):
    ic[i] = icfunc(de.points[i, 0], de.points[i, 1])
ic = ic[num_points:]
ic = concatenate((ic, zeros(len(ic))))


def heat_solver(t, vec):
    return dot(Minv, dot(-Knew, vec))

def wave_solver(t, vec):
    vec_len = len(vec)
    avec = vec[0:int(vec_len/2)]
    vvec = vec[int(vec_len/2):vec_len]
    vvec_dot = dot(Minv, dot(-Knew, avec))
    return(concatenate((vvec, vvec_dot)))

tend = 30
t_pts = 900
t = linspace(0, tend, t_pts)
y1 = solve_ivp(wave_solver, [0, tend], ic, t_eval = t, atol = 1e-9, rtol = 1e-9)
assert y1.success

# Plot
from matplotlib import tri
mytri = tri.Triangulation(de.points[:,0],de.points[:,1],de.simplices)

for i in range(t_pts):
    fig,ax = plt.subplots()
    c = y1.y[0:int(y1.y.shape[0]/2),i]
    c = hstack((zeros(int(num_points)),c))
    ax.tricontourf(mytri, c, levels = linspace(-1,2,40))

    fig.savefig("imgs/fig"+str(i)+".png")
    plt.close(fig)