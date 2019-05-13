
#%%
import os
import pandas as pd
from skimage import io,transform
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
##1.从csv中读取数据

landmarks_frame=pd.read_csv('./faces/face_landmarks.csv')
n=1

img_name=landmarks_frame.iloc[n,0]
landmarks=landmarks_frame.iloc[n,1:].as_matrix()


#%%
img_name


#%%
print(type(landmarks),'\n',landmarks.shape)


#%%
landmarks=landmarks.astype('float').reshape(-1,2)


#%%
landmarks[:3,:]


#%%
def show_img(img_name,landmarks):
    plt.imshow(img_name)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,color='red',marker='.')
    plt.pause(0.01)
    
img=io.imread(os.path.join('./faces',img_name))


#%%
show_img(img,landmarks);


#%%
##2.为faces写一个dataset，便捷读取
class FaceLandmarksDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir
        self.transform=transform
        self.landmark_frame=pd.read_csv(os.path.join(root_dir,csv_file))
       
    def __len__(self):
        return len(self.landmark_frame)
    
    def __getitem__(self,index):
        if index > len(self.landmark_frame):
            raise IndexError('{} out of range{}'.format(index,self.landmark_frame))
            
        img_name=self.landmark_frame.iloc[index,0]
        
        img_file=os.path.join(self.root_dir,img_name)
        img=io.imread(img_file)
        landmarks=self.landmark_frame.iloc[index,1:].as_matrix().astype(float).reshape(-1,2)
        
        sample={'image':img,'landmarks':landmarks}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


#%%
fDataset=FaceLandmarksDataset('./faces','face_landmarks.csv')
sample=fDataset[2]


#%%
show_img(sample['image'],sample['landmarks'])


#%%
for i in range(5):
    sample=fDataset[i]
    print(i,':',sample['image'].shape,'\n')
    ax=plt.subplot(1,5,i+1)
    plt.tight_layout()
    ax.set_title('sample #{}'.format(i))
    ax.axis('off')
    show_img(sample['image'],sample['landmarks'])


#%%
## 3.对图像进行变换

##手动写3个支持可调用的类，以支持：
##>缩放
##>裁剪
##>转换为Tensor

## 3.1 缩放
class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
        
    def __call__(self,sample):
        img,landmarks=sample['image'],sample['landmarks']
        
        h,w=img.shape[:2]
        if isinstance(self.output_size,int):
            if h > w:
                w_new = self.output_size
                h_new = (h/w)*self.output_size
            else:
                h_new = self.output_size
                w_new = (w/h)*self.output_size                
        else:
            h_new,w_new = self.output_size[0],self.output_size[1]
        
       ## print(h_new,w_new)
        img_new = transform.resize(img,(int(h_new),int(w_new)))
        landmarks_new = landmarks * [w_new/w,h_new/h]
        
        return {'image':img_new,'landmarks':landmarks_new}


#%%
res=Rescale(224)
t=res(fDataset[0])


#%%
show_img(t['image'],t['landmarks'])


#%%
## 3.2 随机裁剪
class RandomCrop(object):
    def __init__(self,output_size):
        if not isinstance(output_size,(int,tuple)):
            raise TypeError('only int or tuple type supported.')
        
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            if len(ouput_size) != 2:
                raise ValueError('tupe size should be 2.')
            else:
                self.output_size = output_size
        
        
    def __call__(self,sample):
        img ,landmarks = sample['image'],sample['landmarks']
        
        h,w         = img.shape[:2]
        h_new,w_new = self.output_size[0],self.output_size[1]
        
        top = np.random.randint(0,h-h_new)
        left = np.random.randint(0,w-w_new)
        
        img_new=img[top:top+h_new,left:left+w_new]
        landmarks_new=landmarks - [left,top]
        
        return {'image':img_new,'landmarks':landmarks_new}


#%%
rcrop=RandomCrop(112)
t=rcrop(fDataset[0])


#%%
show_img(t['image'],t['landmarks']);


#%%
## 3.3 ToTensor
class ToTensor(object):
    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        
        image = image.transpose((2,0,1))
        
        return {'image':torch.tensor(torch.from_numpy(image)),'landmarks':torch.tensor(torch.from_numpy(landmarks))}


#%%
scale=Rescale(256)
crop=RandomCrop(128)
composed=transforms.Compose([Rescale(256),RandomCrop(224)])
sample=fDataset[65]

for i,tf in enumerate([scale,crop,composed]):
    t_sample = tf(sample)
    ax = plt.subplot(1,3,i+1)
    plt.tight_layout()
    ax.set_title(type(tf).__name__)
    show_img(t_sample['image'],t_sample['landmarks']);


#%%
transformed_facedataset=FaceLandmarksDataset('./faces','face_landmarks.csv',
                                             transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))
for i in range(4):
    img ,landmarks= transformed_facedataset[i]['image'],transformed_facedataset[i]['landmarks']
    print(i,' image size: ',img.shape,' landmarks size: ',landmarks.shape)


#%%
##4.在上面简单的循环中，没有实现对样本的混淆、批采样、多线程加载等
##使用utils.data.DataLoader可以实现上面的功能

dl=DataLoader(transformed_facedataset,shuffle=True,batch_size=4)
def show_batched_sample(batched_sample):
    img_batched,landmarks_batched = batched_sample['image'],batched_sample['landmarks']
    
    img_sz = img_batched.shape[2]
    batch_sz = len(img_batched)
    
    grid = utils.make_grid(img_batched)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    
    for i in range(batch_sz):
        plt.scatter(landmarks_batched[i,:,0].numpy()+i*img_sz,landmarks_batched[i,:,1].numpy(),s=10,c='r',marker='.')
        
    plt.title('batch from dataloader')
    


#%%
for i,sample_batched in enumerate(dl):
   # print(i,' img batch-sz: ',sample_batched['image'].shape,' landmarks batch-sz: ',sample_batched['landmarks'])
    
    
    if i == 3:
        show_batched_sample(sample_batched)
        plt.axis('off')
        break


#%%
class ToPIL(object):
    def __call__(self,sample):
        img = sample['image']
        toPIL = transforms.ToPILImage()
        return toPIL(img)


#%%
##5.使用trochvision的transform
##由于之前的图像数据使用SKImage读取，而torchvision的transfroms针对的是PIL
##因此要先进行转换
tface=FaceLandmarksDataset('./faces','face_landmarks.csv',
                                             transforms.Compose([ToPIL(),
                                                                 transforms.RandomResizedCrop(224),
                                                                 transforms.RandomHorizontalFlip()]))

type(tface[1])


#%%
tface[1].size


#%%
for i in range(20):
    print(tface[i].size)


#%%



