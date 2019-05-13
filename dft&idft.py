#%% [markdown]
# # Example of Computing DFT&iDFT On Image
# 
#  An example just shows how to compute dft&idft on images to better understand the Discrete Fourier Transform . Without taking any efficiency consideration, to keep it simple and clear.
#  
# Then compare the results with np.fft.fft2.
# 
# ## Formula:
# ### DFT: $F(h,k)=\sum_{a=0}^{N-1} \sum_{b=0}^{N-1} f(a,b)e^{\frac{-j2\pi}{N}(ah+bk)}$
# 
# ### iDFT：$f(a,b)=\frac{1}{N^2}\sum_{h=0}^{N-1} \sum_{k=0}^{N-1} F(h,k)e^{\frac{j2\pi}{N}(ah+bk)}$
# 
# ## DFT Steps:
# - consider f(a,b) as complex number(though it is a real pixel value of 2D image) and compute its magnitude as:$\|f(a,b)\|=np.abs(f(a,b))$, which means for a complex number C:   $magnitude(C)=sqrt\{(C.real)^2+(C.imag)^2\}$
# 
# 
# - compute phase as $\theta = -(\frac{2\pi}{N}(ah+bk))$
# 
# 
# - store the temp complex value as:$F(h,k)_{a,b} = \|f(a,b)\|cos\theta+j\|f(a,b)\|sin\theta$
# 
# 
# - sum the temp to get final:$F(h,k) = \sum_{a=0}^{N-1} \sum_{b=0}^{N-1} F(h,k)_{a,b}$
# 
# ## iDFT Steps:
# almost like DFT Steps.

#%%
import numpy as np
import cv2
import cv_helper as ch
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

img_path=ch.get_img_path('lena')
lena=cv2.imread(img_path,0)
N=lena.shape[0]#here shape[0]==shape[1]
lena_dft=np.fft.fft2(lena)
lena_idft=np.fft.ifft2(lena_dft)

def dft(img,N,h,k):
    F=np.zeros((N,N),dtype=np.complex128)
    w0=2*np.pi/N
    for a in range(N):
        for b in range(N):
            mag=img[a,b]
            theta=-w0*(a*h+b*k)
            F[a,b]=np.complex128(mag*np.cos(theta)+mag*np.sin(theta)*1j)

    return F.sum()

def idft(F,N,a,b):
    I=np.zeros((N,N),dtype=np.complex128)
    w0=2*np.pi/N
    for h in range(N):
        for k in range(N):
            mag=np.abs(F[h,k])
            theta1=np.angle(F[h,k])
            theta2=w0*(a*h+b*k)
            theta=theta1+theta2
            I[h,k]=np.complex128(mag*np.cos(theta)+mag*np.sin(theta)*1j)

    return I.sum()/(N*N)

#test it
print('dft:\n',lena_dft[55,128],'\n',dft(lena,N,55,128))
print('idft:\n',lena_idft[200,255],'\n',idft(lena_dft,N,200,255))


#%%
lena_dft.sum()/(N*N)


#%%
idft(lena_dft,N,0,0)


#%%
lena[0,0]


#%%
x,y=np.meshgrid([0,1,2,3],[0,1,2,3])


#%%
r=np.vstack((x.reshape(1,-1),y.reshape(1,-1)))


#%%
r


#%%
row=r[0,:].copy()
r[0,:]=r[1,:]
r[1,:]=row
r


#%%
r=np.swapaxes(r,0,1).reshape(4,4,2)
r


#%%
ax=plt.axes()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()
ax.scatter(r[:,:,0],r[:,:,1]);


#%%
np.fft.fftshift(r)


#%%
t=r[:,:,0]
r[:,:,0]=r[:,:,1]
r[:,:,1]=t


#%%
r


#%%
get_ipython().run_line_magic('pinfo', 'np.amin')


#%%



