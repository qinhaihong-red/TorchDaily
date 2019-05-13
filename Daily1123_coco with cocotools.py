
#%%
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import os,math
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1.JSON文件初始化

#%%
ann_file='./coco/annotations/instances_train2014.json'
coco=COCO(ann_file)

#%% [markdown]
# ## 2.CatID & CatNames 

#%%
##先获得cat id
cats_id=coco.getCatIds()
cats=coco.loadCats(cats_id)
##再获得id对应的item
cats


#%%
##所有子类别列表
names= [i['name'] for i in cats]
print('all cats:\n{}'.format(' '.join(names)))


#%%
##所有大类别列表
super_cats=set([i['supercategory'] for i in cats])
print('super_cats:\n{}'.format(' '.join(super_cats)))

#%% [markdown]
# ## 3.图像加载

#%%
catsId=coco.getCatIds(catNms=['person','ball'])
imgsId=coco.getImgIds(catIds=catsId)
imgsId[0:2]


#%%
img=coco.loadImgs(imgsId[np.random.randint(0,len(imgsId))])[0]
img


#%%
I=io.imread(os.path.join(img_dir,img['file_name']))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(I);

#%% [markdown]
# ## 4.对象检测点

#%%

annsId=coco.getAnnIds(imgIds=img['id'],catIds=catsId,iscrowd=False)
anns=coco.loadAnns(annsId)
plt.axis('off')
plt.imshow(I)
coco.showAnns(anns)


#%%
x,y,W,H=anns[0]['bbox']


#%%
x_lin=np.linspace(x,x+W)
y_lin=np.linspace(y,y+H)
plt.axis('off')
plt.imshow(I)
x_fixed=np.full((1,len(x_lin)),x).reshape(50,)
y_fixed=np.full((1,len(y_lin)),y).reshape(50,)
plt.plot(x_lin,y_fixed,color='blue')
plt.plot(x_lin,y_fixed+H)
plt.plot(x_fixed,y_lin)
plt.plot(x_fixed+W,y_lin);

#%% [markdown]
# ## 5.关键点检测点

#%%
kp_ann_file='./coco/annotations/person_keypoints_train2014.json'
kp_coco=COCO(kp_ann_file)


#%%
kp_annsId=kp_coco.getAnnIds(imgIds=img['id'],catIds=catsId,iscrowd=False)
kp_anns=kp_coco.loadAnns(kp_annsId)

plt.axis('off')
plt.imshow(I)
kp_coco.showAnns(kp_anns)

#%% [markdown]
# ## 6.说明文字

#%%
cp_ann_file='./coco/annotations/captions_train2014.json'
cp_coco=COCO(cp_ann_file)


#%%
cp_annsId=cp_coco.getAnnIds(imgIds=img['id'])##这里只要imgIds一个关键字就好，其他的不要
cp_anns=cp_coco.loadAnns(cp_annsId)

plt.axis('off')
plt.imshow(I);
cp_coco.showAnns(cp_anns)


#%%



