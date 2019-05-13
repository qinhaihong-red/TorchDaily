
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# 参考：https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# 
# # 0.开始
# 
# 
# ![image](https://pytorch.org/tutorials/_images/FSeq.png)
# 
# 在本篇中，通过使用基于视觉注意机制（visual attentin mechanism）的空间变换网络（Spactial Transformer Network），来学习如何增强网络.
# 
# STN是对任何空间变换可区别注意的一种泛化. 为增强网络的几何不变性，STN允许神经网络学习对输入图像进行空间变换，来达到几何不变性的目的.
# 
# 例如，可以裁剪感兴趣的区域，缩放和纠正图像的方向. 由于CNN对于旋转、缩放以及其他更一般的仿射变换是可变的，因此这就是一种有用的机制.
# 
# 关于STN最棒的其中一点就是，它可以简单插入现有的神经网络，而不用多做修改.
#%% [markdown]
# # 1.数据加载
# 
# 在本篇中，我们使用经典的 MNIST 数据集. 使用具备STN增强的标准神经网络 

#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_dir='../data/mnist/'

process=['train','test']

mnist_transforms=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize([0.1307],[0.3081])])
                                        
mnist_datasets={x:tv.datasets.MNIST(data_dir,train=(x=='train'),download=False,transform=mnist_transforms) for x in process}

mnist_dataloader={x:torch.utils.data.DataLoader(mnist_datasets[x],shuffle=True,batch_size=64) for x in process}
                                        


#%%
len(mnist_datasets['train'])


#%%
len(mnist_datasets['test'])


#%%
item=mnist_datasets['train'][0]


#%%
len(item)


#%%
type(item[0])


#%%
img=item[0]
img.size


#%%
##由于在dataset中进行了正则化，这里需要再逆操作
mean=torch.tensor([0.1307])
std=torch.tensor([0.3081])
for i in range(4):
    ax=plt.subplot(1,4,i+1)
    item=mnist_datasets['train'][i]
    img=item[0]*std+mean
    img=tv.transforms.functional.to_pil_image(img)
    img=tv.transforms.functional.resize(img,size=(224,224))
    ax.set_title(item[1].item())
    plt.imshow(img,cmap='gray');


#%%
##无正则化
for i in range(4):
    ax=plt.subplot(1,4,i+1)
    item=mnist_datasets['train'][i]

    img=tv.transforms.functional.to_pil_image(item[0])
    ax.set_title(item[1].item())
    plt.imshow(img,cmap='gray');


#%%
#批图片
batch,_=next(iter(mnist_dataloader['train']))
batch.shape


#%%
##通过make_grid，通道变成了3
grid=tv.utils.make_grid(batch)
grid=grid.permute(1,2,0)
grid.shape


#%%
##注意，如果是3通道数据，反正则化的时候，通道要放到后面，这样才能广播.
mean=torch.tensor([0.1307,0.1307,0.1307])
std=torch.tensor([0.3081,0.3081,0.3081])
grid=grid*std+mean


#%%
grid_img=tv.transforms.functional.to_pil_image(grid.permute(2,0,1))
plt.imshow(grid_img);


#%%
grid=tv.utils.make_grid(batch)
grid=grid.permute(1,2,0)

mean=torch.tensor([0.485, 0.456, 0.406])
std=torch.tensor([0.229, 0.224, 0.225])
grid=grid*std+mean
grid_img=tv.transforms.functional.to_pil_image(grid.permute(2,0,1))
plt.imshow(grid_img);

#%% [markdown]
# # 2. 对STN的描述
# 
# STN网络主要归结为3部分：
# 
# ![image](https://pytorch.org/tutorials/_images/stn-arch.png)
# 
# - 局部化网络. 该网络是个常规的CNN网络，用于对变换的参数进行回归. 网络自动的从空间变换中学习能增强整体准确率的空间变换，而不是显式的从数据集中学习. 
# 
# 
# - 网格产生器. 从输入图像中生成与输出图像每个像素都对应的坐标网格.
# 
# 
# - 采样器. 采样器使用变换的参数，并作用于输入图像.

#%%
class NetWithSTN(nn.Module):
    def __init__(self,withSTN=True):
        super(NetWithSTN,self).__init__()
        self.withSTN=withSTN
        
        self.features=nn.Sequential(
        nn.Conv2d(1,10,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),          
            nn.Conv2d(10,20,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2)
        )
        
        self.classifier=nn.Sequential(
        nn.Linear(20*7*7,100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100,10)
        )
        
        
        self.localization=nn.Sequential(
        nn.Conv2d(1,10,3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,20,3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
                    
        )
        
        self.localization_fc=nn.Sequential(
            nn.Linear(20*7*7,100),
            nn.ReLU(True),
            nn.Linear(100,6)    
        )
        
        self.localization_fc[-1].weight.data.zero_()
        self.localization_fc[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float))

        
    def STN(self,x):

        local_out=self.localization(x)
        local_out=local_out.reshape(-1,20*7*7)
        theta=self.localization_fc(local_out)
        
        theta=theta.reshape(-1,2,3)
        grid=F.affine_grid(theta,x.size())
        x=F.grid_sample(x,grid)
        
        return x
    
    def forward(self,x):
        if self.withSTN:
            x=self.STN(x)


        x=self.features(x)
        x=x.reshape(-1,20*7*7)
        x=self.classifier(x)
            
        return x

#%% [markdown]
# # 3.训练模型
# 
# 现在使用SGD对模型进行训练. 网络以监督学习的方式学习分类任务.同时模型也以端到端的方式自动学习STN.

#%%
len(mnist_dataloader['train'])*mnist_dataloader['train'].batch_size


#%%
len(mnist_dataloader['train'])


#%%
len(mnist_datasets['train'])


#%%
E=nn.CrossEntropyLoss()
def train(mod,op,epoch,dataloader=None):
    model=mod
    opt=op
    model.train()

    if dataloader is None:
        dataloader=mnist_dataloader
    
    for i,(input,target) in enumerate(dataloader['train']):
        input,target=input.to(device),target.to(device)
        opt.zero_grad()
        out=model(input)
        loss=E(out,target)
        loss.backward()
        opt.step()
        
        if i% 500 == 0:
            progress=(i+1)*dataloader['train'].batch_size
            progress=progress/len(dataloader['train'].dataset)
            print('Train Epoch:{},Progress:{:.1f},Loss:{:.4f}'.format(epoch,progress,loss.item()))


#%%
def test(mod,op,epoch,dataloader=None):
    model=mod
    opt=op
    with torch.no_grad():
        model.eval()
        correct=0
        loss_all=0.
        
        if dataloader is None:
            dataloader=mnist_dataloader
        
        for input,target in dataloader['test']:
            input,target=input.to(device),target.to(device)
            
            out=model(input)
            loss=E(out,target)
            loss_all+=loss.item()*input.shape[0]
            _,pred=torch.max(out,dim=1)
            correct+=torch.sum(pred==target).item()
           
        data_size=len(dataloader['test'].dataset)
        print('Test Loss:{:.4f},Acc:{:.4f}'.format(loss_all/data_size,correct/data_size))
    

#%% [markdown]
# # 可视化STN结果
# 
# 现在可以检测一下所学的视觉注意机制.
# 
# 定义一个帮助函数，来可视化训练中的变换：

#%%
def convert_img_np(img):
    img_np=img.numpy().transpose((1,2,0))
    
    ##反正则化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    
    img_np=img_np*std+mean
    img_np=np.clip(img_np,0,1)
    
    return img_np

#%% [markdown]
# 经过对模型的训练之后，我们打算可视化STN的输出，即：可视化一批输入的图片，以及对应的STN的输出.

#%%
def visualize_stn(mod):
    model=mod
    with torch.no_grad():
        input_img=next(iter(mnist_dataloader['test']))[0].to(device)
        
        ##通过plt展示，需要转回到cpu上
        input_tensor=input_img.cpu()
        tf_input_tensor=model.STN(input_img).cpu()
        
        in_grid=tv.utils.make_grid(input_tensor)
        out_grid=tv.utils.make_grid(tf_input_tensor)
        
        f,axes=plt.subplots(1,2)
        axes[0].imshow(convert_img_np(in_grid))
        axes[0].set_title('Dataset Image')
        axes[1].imshow(convert_img_np(out_grid))
        axes[1].set_title('Transformed Image')
        


#%%
#使用STN
model=NetWithSTN().to(device)
opt=optim.SGD(model.parameters(),lr=0.01)
for e in range(1,4):
    train(model,opt,e)
    test(model,opt,e)

#%% [markdown]
# __使用STN，准确率最高达到98%__

#%%
visualize_stn(model);


#%%
##未使用STN
model2=NetWithSTN(withSTN=False).to(device)
opt2=optim.SGD(model2.parameters(),lr=0.01)
for e in range(1,4):
    train(model2,opt2,e)
    test(model2,opt2,e)

#%% [markdown]
# __未使用STN，最好的结果是96%__

