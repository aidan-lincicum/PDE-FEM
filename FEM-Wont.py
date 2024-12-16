# Heat and wave equation solver
from numpy import *
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay, delaunay_plot_2d
from numpy.linalg import norm, inv
from scipy.sparse.linalg import spsolve
from PIL import Image, ImageOps

# Constants
num_points = 200
xmin = -1.5
xmax = 1.5
ymin = -1.5
ymax = 0

# Create region
fig,ax = plt.subplots()
y0 = -2+sqrt(3.85)
x0 = sqrt(4*y0+2.4)
x = linspace(-x0,x0,300)
ax.plot(x,-0.6+0.25*x**2)
ax.plot(x,-sqrt(1.5**2-x**2))
ax.plot([x0,-x0],[y0,y0],'ro')
ax.set_aspect("equal")

# upper edge
x1 = linspace(-x0,x0,int(num_points/2))
y1 = -0.6+0.25*x1**2

# lower edge
th = linspace(pi,2*pi,int(num_points/2 + 2))
x2 = 1.5*cos(th)
y2 = 1.5*sin(th)
x2 = x2[y2<y0]
y2 = y2[y2<y0]

# grid in interior
xx,yy = meshgrid(linspace(-1.5,1.5,int(num_points/2)),linspace(-2.05,0.95,int(num_points/2)))
xx[::2,:] += 0.5*3/20
keep = logical_and(xx**2+yy**2<1.43**2, yy<-0.66+0.25*xx**2)

x3 = xx[keep].reshape(-1)
y3 = yy[keep].reshape(-1)

xtri, ytri = hstack((x1,x2,x3)),hstack((y1,y2,y3))

# Create triangulation
de = Delaunay(stack((xtri,ytri)).T)
neighbors = de.neighbors
triangles = de.points[de.simplices,:]
centroids = triangles.sum(axis=1)/3
keep = centroids[:,1] < -0.6+0.25*centroids[:,0]**2
de.simplices = de.simplices[keep,:]
triangles = de.points[de.simplices,:]

# Find gradient of the triangles
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

# Create K matrix
pts = len(de.points)
# from scipy.sparse import coo_matrix
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

# Create M matrix
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

# Create image as initial conditions
# im1 = Image.open("imgs/cat.jpg")
# im2 = ImageOps.grayscale(im1)
# numpydata = asarray(im2)
# dim = numpydata.shape
# ic = zeros(pts)
# for i in range(pts):
#     xpos = int(de.points[i,0]/(xmax - xmin)*dim[0] - xmin/(xmax - xmin)*dim[0])
#     ypos = int(de.points[i,1]/(ymin - ymax)*dim[1] - ymin/(ymin - ymax)*dim[1])
#     ic[i] = numpydata[ypos,xpos]
# ic = ic[int(num_points):]
# ic = concatenate((ic, zeros(len(ic))))


# Create function as initial conditions
def icfunc(x, y):
    return x + y
ic = zeros(pts)
for i in range(pts):
    ic[i] = icfunc(de.points[i, 0], de.points[i, 1])
ic = ic[num_points:]
ic = concatenate((ic, zeros(len(ic))))

# a' = M^{-1}(-Ka)
def heat_solver(t, vec):
    return dot(Minv, dot(-Knew, vec))

# v' = M^{-1}(-Ka)
# a' = v
def wave_solver(t, vec):
    vec_len = len(vec)
    avec = vec[0:int(vec_len/2)]
    vvec = vec[int(vec_len/2):vec_len]
    vvec_dot = dot(Minv, dot(-Knew, avec))
    return(concatenate((vvec, vvec_dot)))

# ODE Solver
tend = 30
t_pts = 900
t = linspace(0, tend, t_pts)
y1 = solve_ivp(wave_solver, [0, tend], ic, t_eval = t, atol = 1e-9, rtol = 1e-9)
assert y1.success

# Plot
from matplotlib import tri
mytri = tri.Triangulation(de.points[:,0],de.points[:,1],de.simplices)
# cax = plt.axes((max(c), min(c), 0.075, 0.8))
# plt.colorbar(ax, cax=cax, cmap="summer")
for i in range(t_pts):
    fig,ax = plt.subplots()
    c = y1.y[0:int(y1.y.shape[0]/2),i]
    c = hstack((zeros(int(num_points)),c))
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_trisurf(mytri, c, linewidth=0.2, antialiased=True)
    ax.tricontourf(mytri, c, levels = linspace(-3,3,40))
    #fig.colorbar(pl)
    # ax.set_zlim(-1.5,1)
    fig.savefig("imgs/fig"+str(i)+".png")
    plt.close(fig)