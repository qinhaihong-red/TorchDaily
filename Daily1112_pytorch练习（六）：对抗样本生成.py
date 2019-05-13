
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
import os,math,time
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# 原文：https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# 
# 作者: Nathan Inkawhich
# 
# 翻译：QIN Haihong
# 
# 
# # 对抗样本生成（Adversarial Example Generation）
# 
# 读到这里，希望你能感受到某些机器学习模型是多么有效. 对机器学习的研究，不断推动机器学习模型变得更快、更精确、更有效.
# 
# 然而，在设计和训练模型中，安全性和鲁棒性则属于经常被忽视的方面. 尤其是在面对有意糊弄模型的对抗样本.
# 
# 本教程旨在提高你关于机器学习模型安全漏洞方面的意识，同时进一步了解对抗机器学习这一火热话题.
# 
# 你将会惊奇的发现，为图像添加不可察觉的扰动，将会导致模型彻底不同的变化.
# 
# 考虑到这是个教程，我们将会通过图片分类的示例来探索这一话题.
# 
# 具体来说，我们将会使用最流行的攻击方法之一，快速梯度符号攻击（Fast Gradient Sign Attack，FGSA），来糊弄一个MNIST分类器.
#%% [markdown]
# ## 威胁模型
# 
# 需要指出的是，对抗攻击有很多种类，根据每一种根据攻击者的知识，将会有不同的目标和假设.
# 
# 但是总体上来说，攻击的首要目标是为输入数据加入最小量的扰动，以导致期望的误分类.
# 
# 关于攻击者的知识，有几种假设，其中两个是：白盒与黑盒:
# 
# - 白盒攻击假设攻击者具有模型的全部知识，并能接入模型，包括架构，输入、输出和权重. 
# 
# - 黑盒攻击假设攻击者只能接入模型的输入和输出，对于模型的底层架构或权重却一无所知.
# 
# 同时也存在几种目标，包括误分类和来源/目标误分类：
# 
# - 误分类：它的对抗目标是，使得输出分类错误，但是不关心新的目标是什么.
# 
# - 来源/目标误分类：它的对抗目标是，把实际来源于A类的样本，误分类为特定的B类.
# 
# 
# 在本例中，FGSA是一个以误分类为目标的白盒攻击. 有了这些背景信息，我们可以讨论攻击细节.
#%% [markdown]
# ## FGSA
# 
# FGSA(Fast Gradident Sign Attack)，是迄今为止最流行的对抗攻击之一. 该方法出自 https://arxiv.org/abs/1412.6572 这篇论文，由Goodfellow等人提出.
# 
# 该方法相当有威力，同时也非常直观. 即，通过利用神经网络使用梯度进行学习的方式，来攻击神经网络.
# 
# 这个想法很简单：就是基于相同的反向传播得到的梯度，攻击者调整输入数据，以最大化损失函数，而不是通过基于反向传播得到的梯度，来调整权值，以最小化损失函数.
# 
# __换句话说，攻击者使用损失函数关于输入数据的梯度，来调整输入数据，来最大化损失函数.__
# 
# 在开始编码前，先看看著名的FGSA熊猫，并抽取处一些概念.
#%% [markdown]
# ![image](https://pytorch.org/tutorials/_images/fgsm_panda_image.png)
#%% [markdown]
# $x$是原始输入图像，正确分类为熊猫;
# 
# $y$是$x$的真实标签，$\theta$表示模型的参数，$J(\theta,x,y)$是用来训练框架的损失函数.
# 
# 攻击者把梯度反向传播到输入数据，以计算$\nabla_{x}J(\theta,x,y)$. 
# 
# 然后，在使得损失最大化的方向上（即$sign(\nabla_{x}J(\theta,x,y))$），以较小的步长（$\epsilon$或者图上的0.007）调整输入数据.
# 
# 被扰动的图像结果，$x{'}$，被目标网络误分类为长臂猿，而实际上很明显它仍然是一个大熊猫.
# 
# 下面进行实现.
#%% [markdown]
# ## 实现
# 
# 在本节中，我们将讨论关于本教材的输入参数，定义处于攻击下的模型，然后编码攻击并运行一些测试.
# 
# ### 输入
# 
# 本教程的输入只有3个，定义如下：
# 
# - epsilons: $\epsilon$列表值. 列表中需要有0，因为它表示模型在原始测试集上的表现 . 就降低模型准确率而言，同时也需要更大的$\epsilon$值， 这意味着更可见的扰动，也就是更有效的攻击. 由于这里的数据范围是[0,1]，因此$\epsilon$值不能超过1.
# 
# 
# 
# - 预训练的模型：指向预训练的MNIST模型的路径.
# 
# 
# - use_cuda：是否使用cuda的标志. 

#%%
epsilons=[0,0.05,0.1,0.15,0.2,0.25,0.3]

##这个模型是自定义模型，和原文使用的lenet不同
pretrained_model='./models/customized_mnist_model.pth'
##这里不用use_cuda标志位，直接赋值divice
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#%%
process=['train','val']
norm=tv.transforms.Normalize([0.1307],[0.3081])
transforms=tv.transforms.Compose([tv.transforms.ToTensor()])
dataset_mnist={x:tv.datasets.MNIST('../data/mnist/',train=(x=='train'),download=False,transform=transforms) for x in process}

dataloader={x:torch.utils.data.DataLoader(dataset_mnist[x],shuffle=True,batch_size=64) 
            for x in process}


#%%
class MNISTSub(torch.utils.data.Dataset):
    '''获得MNIST的子集'''
    def __init__(self,mnist,size=10000,transform=None):
        
        self.subset,_=torch.utils.data.random_split(mnist,[size,len(mnist)-size])

        self.size=size
        self.transform=transform
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,i):
        
        img,target=self.subset[i]
            
        if self.transform is not None:
            img=self.transform(img)
            
        return img,target

subsize={'train':50000,'val':5000}
mnist_sub={x:MNISTSub(dataset_mnist[x],subsize[x],transforms) for x in process}
len(mnist_sub['train'])


#%%
##如果认为训练时间过长，可以使用子集进行训练
##实际运行中发现，60000个mnist的训练集，运行4个epochs，不到1分钟就可以训练完毕

subloader={x:torch.utils.data.DataLoader(mnist_sub[x],shuffle=True,batch_size=64) for x in process}

print(len(subloader['train'].dataset),' ',len(subloader['val'].dataset))


#%%
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
        nn.Conv2d(1,10,kernel_size=3,padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10),
            nn.Conv2d(10,20,kernel_size=3,padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20),            
            nn.Conv2d(20,64,kernel_size=3,padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(inplace=True),         
        )
        
        self.classifier=nn.Sequential(
        
            nn.Linear(64*3*3,100),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(100,10),        
        )
      
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.zero_()
            if isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m,nn.Linear):
                m.weight.data.fill_(0.01)
                m.bias.data.zero_()
     
        
    def forward(self,x):
        x=self.features(x)
        x=x.reshape(x.shape[0],64*3*3)
        x=self.classifier(x)
        
        return x

    
model=Net().to(device)
opt=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)##使用动量能提升0.6个百分点，不要把这个参数误写为True
lr_scheduler=optim.lr_scheduler.MultiStepLR(opt,[20,50],0.05)
E=nn.CrossEntropyLoss()


#%%
##注意，train模块可以添加验证集
##在1个epoch中，执行完train，再执行val，这样便于选择最优模型
def train_model(model,dataloader,opt,epoch):
    model.train()
    dataset_len=len(dataloader.dataset)
    begin=time.time()
    best_acc=0.
    for i in range(epoch):
        print('train epoch:{}'.format(i+1))

        correct_all=0.
        loss_all=0.
        for j,(input,target) in enumerate(dataloader):
            input,target=input.to(device),target.to(device)
            
            opt.zero_grad()
            out=model(input)
            loss=E(out,target)
            loss.backward()
            opt.step()
            
            
            loss_all+=loss.item()*input.shape[0]
            _,pred=torch.max(out,dim=1)
            correct_all+=torch.sum(pred==target).item()
            
          
            if j%500==0:

                print('\ttrained :{:.4f}'.format(input.shape[0]*(j+1)/dataset_len)) 

         
        loss=loss_all/dataset_len
        acc=correct_all/dataset_len
            
        print('Loss:{:.4f}, Acc:{:.4f}'.format(loss,acc))
       
        print()
        
    time_elapse=time.time()-begin
    print('time eplapsed:{:.0f}m {:.2f}s'.format(time_elapse//60,time_elapse%60))
    
    
    
if os.path.isfile(pretrained_model):
    model.load_state_dict(torch.load(pretrained_model,map_location='cpu'))
else:
    train_model(model,dataloader['train'],opt,4)
    torch.save(model.state_dict(),pretrained_model)

#%% [markdown]
# ### FGSA 攻击
# 
# 现在可以定义生成对抗样本的函数，该函数通过对原始输入施加扰动来生成样本.
# 
# fgsm_attack函数有3个输入：
# 
# - image表示原始输入$x$
# 
# - $epsilon$表示对图像进行逐像素扰动的量$\epsilon$
# 
# - $data\_grad$表示损失关于输入图像的梯度$\nabla_{x}J(\theta,x,y)$.
# 
# 因此函数生成扰动图像的公式为：
# 
# $perturbed\_image=image+epsilon*sign(data\_grad)=x+\epsilon sign(\nabla_{x}J(\theta,x,y))$
# 
# 最后，为了保持数据的原始范围，经扰动的图像数据需要经裁剪到[0,1]的范围.

#%%
def fgsm_attack(image,e,data_grad):
    dg_sign=data_grad.sign()
    perturbed_img=image+e*dg_sign
    
    perturbed_img=torch.clamp(perturbed_img,0,1)
    
    return perturbed_img

#%% [markdown]
# ### 测试函数
# 
# 最后来到见证最终结果的函数.  每个对此函数的调用，都会对整个MNIST测试集执行一个测试步，并报告最终的精确度.
# 
# 注意函数同时具有一个参数为epsilon的输入，该输入决定了函数中模型处于攻击下的强度，攻击越强，模型报告的准确度越低.
# 
# 具体来说，对于测试集中的每个样本，函数计算损失函数关于输入数据的梯度data_grad，然后使用fgsm_attack生成扰动的图像，
# 
# 然后检测它的对抗性.
# 
# 除了返回模型的精确度，模型同时返回成功生成的对抗样本.

#%%
#测试集的批大小为1
test_loader=torch.utils.data.DataLoader(dataset_mnist['val'],shuffle=True,batch_size=1)
#调整模型到train模式下
model.train()

def test(model,e,test_loader):
    adv_examples=[]

    correct=0.
    ##这个迭代不是为了训练，没有执行优化步，
    ##只是为了计算关于输入的梯度，生成对抗样本.
    for input,target in test_loader:
        input,target = input.to(device),target.to(device)
        
        input.requires_grad_()
        
        out=model(input)
        _,pred=torch.max(out,dim=1)
        
        if pred.item() != target.item():
            continue
            
        loss=E(out,target)
        loss.backward()
        
        perturbed_img=fgsm_attack(input,e,input.grad)#生成对抗样本
        out=model(perturbed_img)
        _,pred2=torch.max(out,dim=1)
        
        if pred2.item() == target.item():
            correct+=1
            if e == 0 and len(adv_examples)<5:
                adv_data=perturbed_img.squeeze(0).detach().cpu().numpy()
                adv_examples.append((pred.item(),pred2.item(),adv_data))
        else:
            if len(adv_examples)<5:
                adv_data=perturbed_img.squeeze(0).detach().cpu().numpy()
                adv_examples.append((pred.item(),pred2.item(),adv_data))
                
    acc=correct/len(test_loader.dataset)
    print('e:{}, Acc={:.0f}/{}={:.4f}'.format(e,correct,len(test_loader.dataset),acc))
    
    return acc,adv_examples

#%% [markdown]
# ### 运行攻击

#%%
epsilons


#%%
acc=[]
adv=[]

for e in epsilons:
    a,ad=test(model,e,test_loader)
    acc.append(a)
    adv.append(ad)

#%% [markdown]
# ## 结果
# 
# ### 精确度 vs Epsilon
# 
# 第一个结果是精确度与e值的对比图. 如先前提到的那样，随着e值增大，可以预期测试精度会下降.
# 
# 这时由于较大的e值意味着我们朝着最大化损失的方向迈的步子更大.
# 
# 注意图中的曲线趋势：尽管e值线性递增，但是精度并不是严格线性降低.
# 
# 例如，e=0.05的精度只比e=0时的精度降低了10%，而e=0.2时的精度比e=0.15时降低了26%.

#%%
plt.figure(figsize=(5,5))
plt.plot(epsilons,acc,'*-')
plt.yticks(np.arange(0,1.1,step=0.1))
plt.xticks(np.arange(0,0.35,step=0.05))
plt.title('Acc VS Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Acc');

#%% [markdown]
# ### 采样对抗样本

#%%
count=0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(adv[i])):
        count+=1
        plt.subplot(len(epsilons),len(adv[0]),count)
        orginal_label,perturbed_label,adv_img=adv[i][j]
        
        img_t=torch.from_numpy(adv_img)
        adv_img=tv.transforms.ToPILImage()(img_t)
        plt.xticks([],[])
        plt.yticks([],[])
        if j==0:
            plt.ylabel('E:{}'.format(epsilons[i]),fontsize=14)
        plt.title('{}->{}'.format(orginal_label,perturbed_label))
        plt.imshow(adv_img,cmap='gray')
        
plt.tight_layout()

#%% [markdown]
# ## 结语
# 
# 希望本教程可以让你了解对抗机器学习. 从这里出发，有很多潜在方向可以探索.
# 
# 这里展示的攻击，属于对抗攻击的最初级手法，已有更多的关于如何攻击和防御机器模型的思路.
# 
# 在NIPS2017上，有个对抗攻击和防御的比赛，其中许多在大赛中使用的方法参见：https://arxiv.org/pdf/1804.00097.pdf.
# 
# 在防御方面进行的工作，使人们思考机器模型需要更大的鲁棒性，无论是对自然的扰动还是对抗的认为的制造的输入数据.
# 
# 另一个对抗攻击和防御的方向是在不同领域内进行考虑.
# 
# 对抗研究从未限制于图像领域，这里有在语言到文字方面的攻击模型：https://arxiv.org/pdf/1801.01944.pdf.
# 

#%%



