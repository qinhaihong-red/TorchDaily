
#%%
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision as tv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 迁移学习 Transfer Learning
# 
# 
# 迁移学习的介绍详见：http://cs231n.github.io/transfer-learning/
# 
# 下面是从中的引用：
# 
# > 实际中，很少有人从头训练一个卷积网络（使用随机的初始化），因为很少具备充足的数据。
# 
# > 比较常见的是，先在一个非常大的数据集（例如，ImageNet包含了1000个类别的120万张图片）训练一个卷积网络，
# 
# > 然后使用该网络对感兴趣的任务做初始化，或者做特征提取.
#%% [markdown]
# 
# 这样两个常用的迁移学习场景，具体如下：
# 
# - 微调网络：使用一个预先训练的网络，如那个在ImageNet训练过的网络，对我们的网络进行初始化，而非随机的初始化. 其余的训练保持例常.
# 
# 
# - 特征提取：除了最后的全链接层，对网络的其他所有层的权值保持固定 。 对最后的全链接层进行替换，替换为一个权值随机初始化的层，并且只对该层进行训练.
#%% [markdown]
# ## 1.数据加载和预处理，可视化
# 
# 
# ### 1.1 训练数据集和验证数据集，目录结构
# 
# 任务是训练一个对蚂蚁和蜜蜂可以正确分类的模型.
# 两者各有120张图片，作为训练集；另外各有75张图片，作为验证集.
# 
# 目录结构：
# - hymenoptera_data
#     - train
#         - ants
#         - bees
#     - val
#         - ants
#         - bees
#%% [markdown]
# ### 1.2 数据加载和预处理
# 
# - 使用transforms进行预处理
#     - 进行裁剪、反转、转张量类型、标准化等
# 
# 
# - 使用Dataset进行数据的读取
#     - 进行预处理、正则化等变换操作
# 
# 
# - 使用DataLoader对Dataset进行批处理
#     - 设置批的大小、混淆以及多线程加载等
#     

#%%
data_transforms = {'train':tv.transforms.Compose([tv.transforms.RandomResizedCrop(224),
                                                  tv.transforms.RandomHorizontalFlip(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
                   'val':tv.transforms.Compose([tv.transforms.Resize(256),
                                               tv.transforms.CenterCrop(224),
                                               tv.transforms.ToTensor(),
                                               tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}


#%%
root_dir='./hymenoptera_data/'
dataset ={x: tv.datasets.ImageFolder(os.path.join(root_dir,x),data_transforms[x]) for x in ['train','val']} 


#%%
dataloader = {x:torch.utils.data.DataLoader(dataset[x],shuffle=True,batch_size=4) for x in['train','val']}


#%%
dataset_sizes = {x:len(dataset[x]) for x in['train','val']}


#%%
dataset_sizes


#%%
class_name = dataset['train'].classes
class_name


#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#%%
device=torch.device('cpu')


#%%
##注意DataLoader是个迭代器
##每次迭代返回的，是根据Dataset来决定的
##这里返回的是个 4维的批Tensor + label
input,label = next(iter(dataloader['train']))
[class_name[i] for i in label]


#%%
input.shape

#%% [markdown]
# ### 1.3 可视化

#%%
def img_show(img_tensor,title=None):
    
    ##torch不支持3d的transpose：
    ##img_tensor=img_tensor.transpose(1,2,0)
    
    ##需要先转回 np
    
    img=img_tensor.numpy()
    img=img.transpose(1,2,0)
    
    mean=np.array([[0.485, 0.456, 0.406]])
    std=np.array([[0.229, 0.224, 0.225]])
    
    img=std*img + mean 
    
    img=np.clip(img,0,1)
    plt.imshow(img);
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)


#%%
grid_img = tv.utils.make_grid(input)
img_show(grid_img,title=[class_name[i] for i in label])


#%%
grid_img.shape


#%%
X=torch.arange(1,7,dtype=torch.float).reshape(3,2)
b=torch.tensor([7,8,9],dtype=torch.float).reshape(3,1)
l=nn.Linear(2,1,bias=False)
E=nn.MSELoss()
print(X,'\n',b)


#%%
loss=E(l(X),b)
loss.item()


#%%
r=l(X)
r


#%%
torch.max(r,1)


#%%
X.max()

#%% [markdown]
# # 2. 训练模型（使用预训练模型）
# 
# __输入：__
# 
# - 模型、损失函数、优化器、学习率调度器(lr)、学习轮数(epoch)
# 
# 
# __输出：__
# 
# - 最优模型；总的损失率；总的正确率；训练时间
# 
# 
# __训练过程：__
# 
# - 最外层对epoch进行循环
# 
# 
# - 每个epoch又分为 train阶段 和 evaluation阶段，即每个epoch内，再进行2个阶段的循环    
#     - train阶段：开启学习率调度器步
#     
# 
# - 在每个阶段内，分别对训练集和验证集，进行批循环：  
#     - train阶段：允许反向传播计算梯度；计算输出与损失；进行反向传播与优化步.
#     
#     - evaluation阶段：禁止反向传播计算梯度；计算输出与损失.
#       
#     - 分别累加每批的损失与正确率
#     
#     - 计算每轮的损失与正确率；记录所有轮数中，最优正确率模型的参数
#      
#      
# - epoch循环结束：
#     - 输出 tain/eval 花费的时间
#       
#     - 返回最优模型
#     
#     
# __要点：__
# - 模型要转到cuda上；dataloader返回的批数据也要转到cuda上
# 
# - 区分train和evaluation，后者不需要反向传播以及优化步

#%%
def train(mod,E,opt,lr_shceduler,epoch_num):
    start = time.time()
    
    best_model_params = copy.deepcopy(mod.state_dict())
    best_acc = 0.0
    
    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch,epoch_num -1 ))
        print('-'*10)
        
        for phase in ['train','val']:
            
            if phase == 'train':
                lr_shceduler.step()
                mod.train()
            else:
                mod.eval()
            
            runing_loss = 0.0
            runing_correct=0
            
            for input,labels in dataloader[phase]:
               
            ##数据要转到GPU上运行
                labels = labels.to(device)
                input = input.to(device)
                        
                
                with torch.set_grad_enabled(phase == 'train'):
                    out=mod(input)
                    loss = E(out,labels)
                    _,preds = torch.max(out,dim=1)
                    
                    if phase == 'train':
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        
                        
                runing_loss    = runing_loss + loss.item()*input.shape[0]
                runing_correct = runing_correct + torch.sum(preds==labels.data) 
                
                
            epoch_loss = runing_loss/dataset_sizes[phase]
            epoch_acc  = runing_correct.double()/dataset_sizes[phase]
            
            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_params = copy.deepcopy(mod.state_dict())
                
        print()
            
    time_eplase=time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_eplase//60,time_eplase%60))
    print('best Acc:{:.4f}'.format(best_acc))

    mod.load_state_dict(best_model_params)

    return mod

#%% [markdown]
# # 3. 对模型预测进行可视化

#%%
def visualize_mod(mod,num_img=6):
    training = mod.training
    mod.eval()
    img_index=0
    
    with torch.no_grad():
        for i,(input,labels) in enumerate(dataloader['val']):
            
            input=input.to(device)
            labels=labels.to(device)
            
            out=mod(input)
            _,preds=torch.max(out,dim=1)
            
            for j in range(input.shape[0]):
                img_index+=1
                ax = plt.subplot(num_img//2,2,img_index)
                ax.axis('off')
                ax.set_title('predicted:{}'.format(class_name[preds[j]]))
                img_show(input.cpu().data[j])
                
                if img_index == num_img:
                    mod.train(mode=training)
                    return
                
    mod.train(mode=training)

#%% [markdown]
# # 4. 微调网络

#%%
device


#%%
device=torch.device('cuda')


#%%
mod=tv.models.resnet18(pretrained=True)
num_features_in = mod.fc.in_features
mod.fc = nn.Linear(num_features_in,2)
mod = mod.to(device)##要转到GPU上运行
E = nn.CrossEntropyLoss()
opt = optim.SGD(mod.parameters(),lr=0.001,momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(opt,step_size=7,gamma=0.1)

#%% [markdown]
# # 5.训练与可视化
#%% [markdown]
# ## 5.1 训练

#%%
mod=train(mod,E,opt,lr_scheduler,10)

#%% [markdown]
# ## 5.2 CPU与GPU训练时间对比
# 
# 使用CPU训练一轮：
# 
# > epoch: 0/0
# > ----------
# >train Loss:0.5539 Acc:0.8033
# >val Loss:0.2541 Acc:0.9150
# 
# >Training complete in __1m 42s__
# >best Acc:0.9150
# 
# 
# 
# 使用GPU训练一轮：
# 
# >epoch: 0/0
# > ----------
# >train Loss:0.5097 Acc:0.7541
# >val Loss:0.3021 Acc:0.8954
# 
# >Training complete in __0m 10s__
# >best Acc:0.8954
#%% [markdown]
# ## 5.3 可视化部分预测结果

#%%
visualize_mod(mod)

#%% [markdown]
# # 6.将网络作为特征提取器
# 
# - 除了最后的全链接层，对于其他层的所有参数，不再计算梯度
#     - 置requires_grad为False
#    
#    
# - 替换fc层，并且把fc的参数传入优化器

#%%
mod2=tv.models.resnet18(pretrained=True)

for param in mod2.parameters():
    param.requires_grad = False
    
fc=nn.Linear(mod2.fc.in_features,2)
mod2.fc=fc


#%%
mod2=mod2.to(device)


#%%
opt = optim.SGD(mod2.fc.parameters(), lr=0.001, momentum=0.9)##只训练全链接层
lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
train(mod2,E,opt,lr_scheduler,10)


#%%



