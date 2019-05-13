
#%%
import numpy as np
import cv2
import cv_helper as ch
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
img_path=ch.get_img_path('lena')
lena=cv2.imread(img_path,0)
ch.imshow_gray(lena)


#%%
lena.shape


#%%
#dft
lena_dft=np.fft.fft2(lena)
#compute mag and log it
lena_dft_mag_log=np.log(np.abs(lena_dft)+1)
#shift it
lena_dft_mag_log=np.fft.fftshift(lena_dft_mag_log)
ch.imshow_gray(lena_dft_mag_log)


#%%
np.all(lena_dft[256:,256:]==np.fft.fftshift(lena_dft)[0:256,0:256])


#%%
lena_dft_angle=np.angle(lena_dft)
lena_dft_angle


#%%
#idft
lena_idft=np.fft.ifft2(lena_dft)
lena_idft_mag=np.abs(lena_idft).astype(np.uint8)#or normalize it
ch.imshow_gray(lena_idft_mag)


#%%
lena_dft[0,0]


#%%
np.power(512,2)


#%%
print(lena[0,0],(lena_dft[0,0].real)/(lena.shape[0]*lena.shape[1]))


#%%
lena.sum()


#%%
N2=lena.shape[0]*lena.shape[1]
real_sum=lena_dft.real.sum()
imag_sum=lena_dft.imag.sum()
a0=real_sum/N2
b0=imag_sum/N2
print(a0,b0)


#%%
print(lena_idft[0,0].real,lena_idft[0,0].imag,np.abs(lena_idft[0,0]))


#%%
#not exactly same
(lena_idft_mag != lena).sum()


#%%
######oepncv#########
#real output
cv_dft=cv2.dft(lena.astype(np.float32))


#%%
out=cv2.normalize(cv_dft,None,0,1,cv2.NORM_MINMAX)
out=np.fft.fftshift(out.astype(np.uint8))
(out==1).sum()


#%%
##complex output
##without log
cv_dft_cplx=cv2.dft(lena.astype(np.float32),cv2.DFT_COMPLEX_OUTPUT)
out=cv2.normalize(np.abs(cv_dft_cplx),0,255,cv2.NORM_MINMAX)
out=np.fft.fftshift(out.astype(np.uint8))


#%%
cv_idft=cv2.idft(cv_dft_cplx)##no effects:flags=cv2.DFT_COMPLEX_OUTPUT
out=cv2.normalize(cv_idft,0,1,cv2.NORM_MINMAX)##or: divide by N2
ch.imshow_gray(out)


#%%
cv_dft[0,0]


#%%
cv_idft[0,0]/N2


#%%
##############################
dog_path=ch.get_img_path('L3')
dog=cv2.imread(dog_path,0)
ch.imshow_gray(dog)


#%%
dog_dft=np.fft.fft2(dog)
dog_dft_shift=np.fft.fftshift(dog_dft)


#%%
dog_dft.shape


#%%
dog_dft_shift.shape


#%%
dog_dft_ishift=np.fft.ifftshift(dog_dft_shift)


#%%
dog_dft_ishift.shape


#%%
np.all(dog_dft_ishift==np.fft.fftshift(dog_dft_shift))


#%%
np.all(dog_dft_ishift==dog_dft)


#%%
np.all(dog_dft==np.fft.fftshift(dog_dft_shift))


#%%
##even shape(both width and height are even):dft=fftshift(fftshift(dft)) == ifftshift(fftshift(dft))
##shpae with an least odd side:dft==ifftshift(fftshift(dft))！= fftshift(fftshift(dft))
freqs = np.fft.fftfreq(16, d=1./9).reshape(4,4)
freqs


#%%
s=np.fft.fftshift(freqs)
s


#%%
np.fft.fftshift(s)


#%%
np.fft.ifftshift(s)


#%%
c=np.zeros((2,3),dtype=np.complex64)
c.real=np.arange(1,7).reshape(2,3)
c.imag=np.random.randn(2,3)


#%%
c


#%%
np.arctan(-1)

#%% [markdown]
# # Compute DFT&iDFT Manually
# 
#  A simple example just shows how to compute dft&idft manually to better understand the Fourier Transform without any efficiency consideration. Then compare the results with np.fft.fft2.
# 
# ## Formula:
# ### DFT: $F(h,k)=\sum_{a=0}^{N-1} \sum_{b=0}^{N-1} f(a,b)e^{\frac{-j2\pi}{N}(ah+bk)}$
# 
# ### iDFT：$f(a,b)=\sum_{h=0}^{N-1} \sum_{k=0}^{N-1} F(h,k)e^{\frac{j2\pi}{N}(ah+bk)}$
# 
# ## DFT Step:
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
# ## iDFT Step:
# almost like DFT Step.

#%%
img_path=ch.get_img_path('lena')
lena=cv2.imread(img_path,0)
N=lena.shape[0]#here shape[0]==shape[1]

def dft(img,N,x,y):
    F=np.zeros((N,N),dtype=np.complex128)
    w0=2*np.pi/N
    for a in range(N):
        for b in range(N):
            mag=img[a,b]
            theta=-w0*(a*x+b*y)
            F[a,b]=np.complex128(mag*np.cos(theta)+mag*np.sin(theta)*1j)

    return F.sum()

def idft(F,N,a,b):
    I=np.zeros((N,N),dtype=np.complex128)
    w0=np.pi/N
    for h in range(N):
        for k in range(N):
            mag=np.abs(F(h,k))
            theta1=np.arctan(F(h,k).imag/F(h,k).real)
            theta2=w0*(a*h+b*k)
            theta=theta1+theta2
            I[h,k]=np.complex128(mag*np.cos(theta)+mag*np.sin(theta)*1j)

    return I.sum()

def get_dft(img,N):
    F=np.zeros((N,N),dtype=np.complex128)
    for x in range(N):
        for y in range(N):
            F[x,y] = dft(img,N,x,y)
    
    return F

#_dft=get_dft(lena,N)
#dft(lena,N,0,1)


#%%



#%%
def print_dft(x,y):
    print('manual-dft results:',dft(lena,N,x,y))
    print('\nfft-dft results:',lena_dft[x,y])
    
print_dft(1,1)


#%%
print_dft(1,0)


#%%
np.pi


#%%
c=np.zeros(4,dtype=np.complex128).reshape(2,2)
c


#%%
c.sum()


#%%
c


#%%
c.real


#%%
c.imag


#%%
c[0,0]+c[1,1]


#%%
np.arctan(c[0,0].imag/c[0,0].real)


#%%
np.angle(c[0,0])


#%%
c.sum()/10


#%%



