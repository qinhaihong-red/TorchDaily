
#%%
import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
root_dir='./coco/images/train2014/'
ann_file='./coco/annotations/instances_train2014.json'
coco=tv.datasets.CocoDetection(root_dir,ann_file)


#%%
len(coco)


#%%
info=coco[0][1]
print(len(coco[0]),' ',type(info),' ',len(info))


#%%
info


#%%
coco[0][0]


#%%
len(coco[0])


#%%
type(coco[0][0])


#%%
len(coco[0][1])


#%%
type(coco[0][1])


#%%



#%%
seg=coco[0][1][0]['segmentation']
seg=seg[0]
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(coco[0][0]);


#%%
seg=coco[0][1][0]['segmentation']
seg=seg[0]
plt.figure(figsize=(10,8))
seg_x=seg[0::2]
seg_y=seg[1::2]
len(seg_y)


#%%
seg=coco[0][1][5]['segmentation']
seg=seg[0]

seg_x=seg[0::2]
seg_y=seg[1::2]
len(seg_y)
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(coco[0][0])
plt.plot(seg_x,seg_y,color='yellow');


#%%



