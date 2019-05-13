
#%%
import numpy as np
import cv2

#%% [markdown]
# # Image Derivative (Part 1)
# 
# ## Theory
# 
# An image can be represented as a mapping : $f(x,y)$, in which $(x,y)$ defines coordinates and $f(x,y)$ for intensity value.
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
# $kx_{1}=\begin{bmatrix}
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
# $kx_{2}=\begin{bmatrix}
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
# $kxy=\begin{bmatrix}
#  1&0&-1\\
#  0&0&0\\
#  -1&0&1\\
# \end{bmatrix}$
# 
# 
# 
# You can get the kernel for $y$ in the same way.

#%%
x,y=cv2.getDerivKernels(1,0,3)
kx1=y*x.transpose()
print(kx1)
print()

x,y=cv2.getDerivKernels(2,0,3)
kx2=y*x.transpose()
print(kx2)
print()

x,y=cv2.getDerivKernels(1,1,3)
kxy=y*x.transpose()
print(kxy)
print()


#%%
idx=cv2.ocl_Device()
idx


#%%
get_ipython().set_next_input('ert=cv2.ocl_Device.vendorName');get_ipython().run_line_magic('pinfo', 'cv2.ocl_Device.vendorName')


#%%
ert=cv2.ocl_Device.vendorName


#%%
import torch


#%%
torch.cuda.is_available()


#%%
cv2.ocl.haveOpenCL()


#%%
cv2.cuda.getCudaEnabledDeviceCount()


#%%
python.__version__


#%%
import sys
sys.version


#%%



