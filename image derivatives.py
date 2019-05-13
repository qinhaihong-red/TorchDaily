
#%%
import numpy as np
import cv2

#%% [markdown]
# # Image Derivative
# 
# ## Theory
# 
# A image can be represented as a mapping : $f(x,y)$, in which $(x,y)$ defines coordinates and $f(x,y)$ for intensity value.
# 
# ### First- and Second Order Derivative
# 
# Let $f(x+\Delta x,y)$ expand as Taylor series:
# ### $f(x+\Delta x,y) = f(x,y)+\dfrac{\partial f}{\partial x}\Delta x+ o(\|\Delta x\|)$.
# 
# Keep the linear terms and let $\Delta x$ be 1,we will get:
# ### $\dfrac{\partial f}{\partial x}=f(x+1,y)-f(x,y)$.
# That's the first order derivative in x direction.
# 
# Taking derivative is a linear transformation.To calculate the second order derivative with respect to $x$,we can apply $\dfrac{\partial f}{\partial x}$ on $\dfrac{\partial f}{\partial x}$ again:
# 
# ### $\dfrac{\partial^{2} f}{\partial x^{2}} = f(x+2,y)-f(x+1,y)-f(x+1,y)+f(x)$.
# 
# Substitute $x$ for $x+1$ because we want the derivative at $x$,we will have:
# 
# ### $\dfrac{\partial^{2} f}{\partial x^{2}} = f(x+1,y)+f(x-1,y)-2f(x,y)$. 
# That's the second order derivative in x direction.
# 
# 
# Using the same approach, we can calculate the first- and second order derivative in y direction, and the mixed partial derviative with respect to  $x$ and $y$.
# 
# We can conclude the derivatives talked above as below:
# 
# - First Order Derivative:
#     - in x direction:$\dfrac{\partial f}{\partial x}=f(x+1,y)-f(x,y)$
#     
#     - in y direction:$\dfrac{\partial f}{\partial y}=f(x,y+1)-f(x,y)$
# 
# - Second Order Derivative:
#     - in x direction:$\dfrac{\partial^{2} f}{\partial x^{2}} = f(x+1,y)+f(x-1,y)-2f(x,y)$
#     
#     - in y direction:$\dfrac{\partial^{2} f}{\partial y^{2}} = f(x,y+1)+f(x,y-1)-2f(x,y)$
#     
#     - mixed partial:$\dfrac{\partial^{2} f}{\partial x \partial y}=f(x+1,y+1)+f(x,y)-f(x+1,y)-f(x,y+1)$
# 
# ### Hessian Matrix
# 
# We can denote the Hessian matrix ,which is based on the second derivative talked above, of $f$ as:
# 
# ##  $H=\begin{bmatrix}
# \dfrac{\partial^{2} f}{\partial x^{2}}&\dfrac{\partial^{2} f}{\partial x \partial y}\\
# \dfrac{\partial^{2} f}{\partial x \partial y}&\dfrac{\partial^{2} f}{\partial y^{2}}
# \end{bmatrix}$
# 
# 
# and we have:
# - eigenvalues：$\lambda^{2}-trace(H)+det(H)=0$,  $\lambda=\dfrac{1}{2}\{trace(H) \pm \sqrt{{trace(H)}^2-4 * det(H)}\}$
# 
# - Laplacian Operator：$\Delta f=trace(H)$
# 
# Hessian matrix is usefull when applied in feature detection such as corners and edges.
# 
# 
# # Practise
# 
# In practise, we usaully convolve the image with specified kernels such as Sobel kernel to approximate the derivative talked above.
# 
# Let's see how these kernels look like and we will use OpenCV to illustrate this.
# 
# Sobel first order kernel for first order derivative approximation in x direction:
# 
# ```
# x,y=cv2.getDerivKernels(1,0,3)
# kx1=y*x.transpose()
# print(kx1)
# ```
# which is: $kx_{1}=\begin{bmatrix}
# -1&0&1\\
# -2&0&2\\
# -1&0&1\\
# \end{bmatrix}$
# 
# 
# second order kernel in x direction:
# ```
# x,y=cv2.getDerivKernels(2,0,3)
# kx2=y*x.transpose()
# print(kx2)
# ```
# whic is :$kx_{2}=\begin{bmatrix}
#  1&-2&1\\
#  2&-4&2\\
#  1&-2&1\\
# \end{bmatrix}$
# 
# 
# and the mixed partial:
# ```
# x,y=cv2.getDerivKernels(1,1,3)
# kxy=y*x.transpose()
# print(kxy)
# ```
# whic is :$kxy=\begin{bmatrix}
#  1&0&-1\\
#  0&0&0\\
#  -1&0&1\\
# \end{bmatrix}$
# 

#%%
import sys
sys.path.append('../')
import cv_helper
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
def dxx(img ,sx = 1.0):
    return sx**2*(img[2:] + img[:-2] - 2*img[1:-1])
 
def dyy(img ,sy = 1.0):
    return sy**2*(img[:,2:] + img[:,:-2] - 2*img[:,1:-1])
 
def dxy(img ,sx = 1.0, sy = 1.0):
    return sx*sy*(img[1:,1:] + img[:-1,:-1] - img[1:,:-1] - img[:-1,1:])
 
def laplace(img, sx =1.0, sy = 1.0):
    return (dxx(img, sx)[:,:-2] + dyy(img, sy)[:-2,:])
 
def eigenOfHessian(img, sx = 1.0, sy = 1.0):
    lam = numpy.empty([2, img.shape[0]-2, img.shape[1]-2])
    H11, H12, H21, H22 = dxx(img, sx)[:,:-2], dxy(img,sx,sy)[:-1,:-1], dxy(img, sx,sy)[:-1,:-1], dyy(img, sy)[:-2,:]
    A, B, C = 1, -(H11 + H22), H11*H22 - H12*H21
    de = numpy.sqrt(B**2 - 4*A*C)
    lam1 = (-B + de)/(2*A)
    lam2 = (-B - de)/(2*A)
    lam = lam1, lam2


#%%
img_path=cv_helper.get_img_path('dog')
img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
cv_helper.imshow_gray(img)


#%%
img.shape


#%%
dx2=dxy(img)
cv_helper.imshow_gray(dx2)


#%%
get_ipython().set_next_input('sobel_x2=cv2.Sobel');get_ipython().run_line_magic('pinfo', 'cv2.Sobel')


#%%
sobel_x2=cv2.Sobel


#%%
sobel_x2=cv2.Sobel(img,cv2.CV_16S,2,2)
cv_helper.imshow_gray(sobel_x2)


#%%
sobel_x2.shape


#%%
sobel_x2.dtype


#%%
M=np.array([1, 2, 5,
4, 5, 11,
7, 8, 9],dtype=np.float32).reshape(3,3)
M


#%%
dxx(M)


#%%
dyy(M)


#%%
cv2.Sobel(M,cv2.CV_32F,1,0)


#%%
cv2.Sobel(M,cv2.CV_32F,2,0)


#%%
cv2.Sobel(M,cv2.CV_32F,0,1)


#%%
cv2.Sobel(M,cv2.CV_32F,2,0).sum()


#%%
cv2.Sobel(M,cv2.CV_32F,0,2).sum()


#%%
cv2.Laplacian(M,cv2.CV_32F,None,3)


#%%
cv2.cornerEigenValsAndVecs(M,3,3)[1,1]


#%%
x,y=cv2.getDerivKernels(1,0,3)
kx1=y*x.transpose()
kx1


#%%
x,y=cv2.getDerivKernels(0,1,3)
ky1=y*x.transpose()
ky1


#%%
x,y=cv2.getDerivKernels(2,0,3)
kx2=y*x.transpose()
kx2


#%%
x,y=cv2.getDerivKernels(0,2,3)
ky2=y*x.transpose()
ky2


#%%
x,y=cv2.getDerivKernels(1,1,3)
kxy=y*x.transpose()
kxy


#%%
print(x,'\n',y)


#%%
cv2.filter2D(M,cv2.CV_32F,kxy)


#%%
cv2.Sobel(M,cv2.CV_32F,1,1)


#%%
H1=np.array([12,-2,-2,-8],dtype=np.float32).reshape(2,2)
H1


#%%
H2=np.array([-36,-2,-2,24],dtype=np.float32).reshape(2,2)
H2


#%%
w1,v1=np.linalg.eig(H1)
w2,v2=np.linalg.eig(H2)


#%%
w1


#%%
w2


#%%
cv2.cornerEigenValsAndVecs(M,1,3)[1,1]


#%%
B=np.ones(9,dtype=np.float32).reshape(3,3)
B[0,0]=0
B[0,1]=7
B[1,1]=5
B


#%%
cv2.boxFilter(B,cv2.CV_32F,(1,1),normalize=False)


#%%



#%%
Dx=cv2.Sobel(M,cv2.CV_32F,1,0,scale=0.000326797)
Dy=cv2.Sobel(M,cv2.CV_32F,0,1,scale=0.000326797)
Dx


#%%
Dy


#%%



