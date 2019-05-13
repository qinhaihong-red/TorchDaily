#%% [markdown]
# __注意:__
# 
# 使用预训练模型有如下事项需要注意:
# 
# - 所有输入的图像数据格式必须为NCHW，HW至少应该是224，C为3. 因此对于MNIST需要在尺寸和通道上进行变换，以符合要求.对于单个数据，要增加一个批的维度.(.unsqueeze(0))
# 
# 
# - 进行正则化的均值和方差统一是：mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225].
# 
# 
# - 由于MNIST数据集比较大（训练集6w，测试集1w，并且由单通道的28x28缩放到3通道的224x224）,需要合理选择子集和批大小，否则造成GPU内存消耗殆尽.
# 
# __结论：__
# 
# 为使用预训练模型，需要对数据集进行变换.
# 
# 变换后增大了数据的量级：从 1x28x28=784 ，增大到 3x224x224=15028，增大了192倍. 极大影响学习效率.

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_dir='../data/mnist/'

process=['train','test']

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
        
        if i% 32 == 0:
            progress=(i+1)*dataloader['train'].batch_size
            progress=progress/len(dataloader['train'].dataset)
            print('Train Epoch:{},Progress:{:.1f},Loss:{:.4f}'.format(epoch,progress,loss.item()))
            
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
    
def channel_transform(x):
    '''把单通道数据转为3通道'''
    assert x.shape[0]==1
    x_new=x.new(3,x.shape[1],x.shape[2])
    x_new[0]=x_new[1]=x_new[2]=x[0]
    ##now shape is：3 H W
    return x_new

#mean=torch.tensor([0.1307,0.1307,0.1307])
#std=torch.tensor([0.3081,0.3081,0.3081])

##使用模型预定义的正则化参数
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transforms_res=tv.transforms.Compose([
    tv.transforms.Resize((224,224)),
    tv.transforms.ToTensor(),
    channel_transform,
    tv.transforms.Normalize(mean,std)   
])


dataset_res={x:tv.datasets.MNIST(data_dir,download=False,train=(x=='train'),transform=transforms_res) for x in process}
dataset_sub={}
dataset_sub['train']=torch.utils.data.Subset(dataset_res['train'],[x for x in range(2000)])
dataset_sub['test']=torch.utils.data.Subset(dataset_res['test'],[x for x in range(200)])
dataloader_res={x:torch.utils.data.DataLoader(dataset_sub[x],shuffle=True,batch_size=8) for x in process}


#%%
print(len(dataset_res['train']),' ',len(dataset_res['test']))


#%%
print(len(dataset_sub['train']),' ',len(dataset_sub['test']))


#%%
model=None
model=tv.models.resnet18(pretrained=True)
model.fc.bias.data[:2]

in_features=512
fc=nn.Linear(512,10)
##先替换fc，再移动到GPU上
model.fc=fc
model=model.to(device)


opt=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
for i in range(1,4):
    train(model,opt,i,dataloader_res)
    test(model,opt,i,dataloader_res)    


#%%



