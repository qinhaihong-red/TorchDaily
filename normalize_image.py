
#%%
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import random


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


#%%
import scipy.misc


#%%
lena = scipy.misc.face()
img = transforms.ToTensor()(lena)
print(img.size())


#%%
imglist = [img, img, img, img.clone().fill_(-10)]


#%%
show(make_grid(imglist, padding=100))


#%%
show(make_grid(imglist, padding=100, normalize=True))


#%%
show(make_grid(imglist, padding=100, normalize=True, range=(0, 1)))


#%%
show(make_grid(imglist, padding=100, normalize=True, range=(0, 0.5)))


#%%
show(make_grid(imglist, padding=100, normalize=True, scale_each=True))


#%%
show(make_grid(imglist, padding=100, normalize=True, range=(0, 0.5), scale_each=True))


#%%



