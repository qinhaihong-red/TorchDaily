
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import os,copy
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ### 总结说明：
# 
# __目标__：产生与内容相似，同时也与风格图像相似的一张图像。这里的相似用距离来度量。这样目标就是极小化输入与两者的距离之和。
# 
# __距离度量__：内容损失使用输入与内容的MSE度量；风格损失使用输入的gram与风格的gram的MSE度量。
# 
# 
# __模型__：采用预训练的vgg19模型的前5层：第4层的卷积后面首先跟着一个内容损失层，再跟着一个风格损失层；其余各层的卷积后面只跟着一个风格损失层。
# 
# __优化器__：optim.LBFGS，优化参数是输入图像。输入图像就是内容图像的副本。
# 
# 
# 
# 
# # 介绍
# 
# 本篇解释如何实现神经风格转移算法，该算法由 Leon A. Gatys, Alexander S. Ecker 和 Matthias Bethge 提出.
# 
# 所谓神经风格转移，就是把一张原始图像，以新的艺术风格的形式再现.
# 
# 该算法取3张图像，一张输入图像(输入图像就是内容图像的复制)，一张内容图像，一张风格图像，通过组装内容图像和风格图像，把输入图像变成具有艺术风格的内容图像.
#%% [markdown]
# # 底层原理
# 
# 原理很简单，定义两组距离，一组关于内容的距离 $D_{C}$，一组关于风格的距离$D_{S}$. 
# 
# $D_{C}$度量两张图像之间内容的不同程度. $D_{S}$度量两种图像之间风格的不同程度.
# 
# 然后，取第3张图像作为输入，对其进行变换，以同时极小化输入图像与内容图像之间的内容距离，和输入图像与风格图像之间的风格距离.
#%% [markdown]
# # 加载图像
# 
# 现在引入风格图像和内容图像.
# 
# 原始的PIL图像像素值位于[0,255]之间，当变换到torch张量以后，像素值就变为了[0,1]之间. 为得到相同的维度，图像同样需要调节大小. 
# 
# 一个重要的细节需要注意，torch库中训练得到的神经网络框架，使用的张量值都是[0,1]之间的. 如果试图给网络输入[0,255]的张量图像，
# 
# 那么激活的特征图将不能感知到预期的内容和风格.
# 
# 作为对比，Caffe库中的预训练网络，使用的是[0,255]的张量图像.

#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
img_size= 512 if torch.cuda.is_available() else 128

img_transformer=tv.transforms.Compose([
    tv.transforms.Resize(img_size),
    tv.transforms.ToTensor()
])

def img_loader(img_file):
    img=Image.open(img_file)
    img_tensor=img_transformer(img).unsqueeze(0)##增加一个批的维度
    return img_tensor.to(device)

img_style=img_loader('./data/picasso.jpg')
img_content=img_loader('./data/dancing.jpg')


#%%
img_content.shape


#%%
def img_show(img,title=None):
    ##要先转移到cpu上
    img=img.cpu().clone()
    img=img.squeeze(0)
    img=tv.transforms.functional.to_pil_image(img)
    plt.imshow(img)
    if title is not None:
        plt.title(title);


#%%
img_show(img_style)


#%%
img_show(img_content)

#%% [markdown]
# # 损失函数
# 
# ## 内容损失
# 
# 内容损失函数，是内容距离的加权版本，为单个层而设计.
# 
# 假设网络当前处理的输入图像是$X$，第$L$层的特征图是$F_{XL}$，该函数以$F_{XL}$作为输入，返回输入图像$X$和内容图像$C$之间的加权内容距离
# $w_{CL}D_{C}^{L}(X,C)$.
# 
# 为计算内容距离，内容图像的特征图$F_{CL}$必须已知.
# 
# 我们以一个torch模块来实现这个函数，它有一个以$F_{CL}$作为输入的构造函数.
# 
# 距离$\|F_{XL}-F_{CL}\|^{2}$ 表示两个特征图集合之间的均方误差，可以使用nn.MSELoss进行计算.
# 
# 当卷积层计算完内容距离之后，我们直接把内容损失模块添加到卷积层后面.
# 
# 当每次给网络输入图像时，内容损失随之也在期望的层中进行计算，由于auto grad机制，所有的梯度都会被自动计算.
# 
# 为了使得内容计算层是透明的，我们定义一个forward方法，来计算内容损失，然后把结果返回给层的输入.
# 
# 已计算的损失，作为模块的参数进行保存.

#%%
class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target=target.detach()
    
    ##计算内容特征图和输入特征图损失的函数
    ##损失保存在loss属性中，仍然返回x，因此该层对于前后层来说，都是透明的
    def forward(self,x):
        self.loss=nn.functional.mse_loss(x,self.target)
        return x

#%% [markdown]
# ## 风格损失
# 
# 风格损失模块的实现，与内容损失模块的实现类似.它计算某层的风格损失，并且对该层而言是透明的. 为了计算风格损失，我们需要计算gram矩阵. 
# 
# 一个gram矩阵，是一个给定矩阵与其转置相乘的结果.
# 
# 在本应用中，给定的矩阵是$L$层的经过重整的特征图$F_{XL}$. 
# 
# $F_{XL}$被重整为$\hat{F}_{XL}$ ，后者是个$K \times N$的矩阵，其中
# K是$L$层特征图的数量，$N$是任何向量化特征图$F_{XL}^{k}$的长度. 
# 
# 例如，$\hat{F}_{XL}$的第一行对应第一个向量化的特征图$F_{XL}^{1}$.
# 
# 最后，gram矩阵必须被正则化：每个矩阵元素除以矩阵中的元素总数. 
# 
# 这个正则化是为了抵消当$\hat{F}_{XL}$具有较大的N维时，产生的gram矩阵中的元素值较大的问题.
# 
# 因为这些较大的值将会导致第一层（在池化层之前）在梯度下降中，具有较大的影响. 由于风格特征趋向于在网络更深层处，因此正则化非常关键.

#%%
def gram_matrix(input):
    a,b,c,d=input.size()
    
    features=input.reshape(a*b,c*d)
    G=torch.mm(features,features.t())
    
    return G.div(a*b*c*d)


#%%
class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target=gram_matrix(target_feature).detach()
    
    ##计算风格特征图和输入特征图损失的函数
    ##损失保存在loss属性中，仍然返回x，因此该层对于前后层来说，都是透明的
    def forward(self,x):
        G=gram_matrix(x)
        self.loss=nn.functional.mse_loss(G,self.target)
        return x

#%% [markdown]
# # 引入模型
# 
# 现在引入预训练过的模型，我们使用19层的vgg模型，就像论文中写的那样.
# 
# vgg模块的pytorch实现，可分为两个子Sequential模块：features模块(包括卷积层和池化层)，和classifier模块(包括全连接层).
# 
# 为了得到每个卷积层的输出，以便于衡量内容和风格的损失，我们需要利用features模块.
# 
# 由于一些层在训练模式的行为与在评估模式的行为有所不同，因此必须使用.eval()方法把模型置于评估模式中.

#%%
#使用的模型：预训练过的vgg19的features部分
#这里优化的目标，不再是模型的参数，而是输入的图像，
#使得输入的图像，经过优化，与内容图像的“距离”较小，与风格图像的“距离”也较小.
model=tv.models.vgg19(pretrained=True).features.to(device).eval()

#%% [markdown]
# 另外，vgg网络是在以如下均值和方差正则化过的图像上进行训练的，所以我们同样需要对图像进行正则化：

#%%
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)


#%%
class Norm(nn.Module):
    def __init__(self,mean,std):
        super(Norm,self).__init__()
        
        ##把均值和方差重整为 3x1x1
        ##这样可以直接与图像张量NCHW进行广播
        self.mean=torch.tensor(mean).reshape(3,1,1)
        self.std=torch.tensor(std).reshape(3,1,1)
        
    def forward(self,x):
        return (x-self.mean)/self.std

#%% [markdown]
# 我们需要把内容损失和风格损失加到卷积层后面. 因此要创建一个新的Sequential模块，以正确添加内容损失和风格损失.
# 
# 下面这个函数演示了如何__拆解并利用已训练模型中部分层__的方法.

#%%
content_layers=['conv_4']
style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_module_and_losses(cnn,mean,std,img_style,img_content,c_layers=content_layers,s_layers=style_layers):
    cnn=copy.deepcopy(cnn)
    norm=Norm(mean,std).to(device)
    
    content_losses=[]
    style_losses=[]
    
    ##先把norm塞进来做正则化
    model=nn.Sequential(norm)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            ##序号只随着卷积层而累加
            ##vgg19=16个卷积层+3个全连接层，其它层如maxpool和relu属于从属层，因此不算
            
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)##ReLU这里处理的不同
        elif isinstance(layer,nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name='bn_{}'.format(i)
        else :
            raise RuntimeError('Unrecognized layer:{}'.format(layer.__class__.name))
            
        model.add_module(name,layer)
        
        if name in c_layers:
            ##在第4个卷积层时，记录内容的卷积输出特征图作为目标
            ##并添加内容损失层
            target=model(img_content).detach()
            content_loss=ContentLoss(target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)
            
        if name in s_layers:
            ##在1，2，3，4，5层，记录风格卷积输出特征图作为目标
            ##并添加风格损失层
            target_feature=model(img_style).detach()
            style_loss=StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)
    
    ##迭代区间[len-1,0]，从后往前迭代，每次减 1
    ##找到最后一个内容损失层或者风格损失层为止
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],(ContentLoss,StyleLoss)):
            break
    
    ##i以后的层都丢弃
    model=model[:(i+1)]
    
    ##返回训练用的model，以及风格损失层列表和内容损失层列表
    ##这个model是对于vgg的拆解并添加了内容损失和风格损失
    return model,style_losses,content_losses

#%% [markdown]
# model输出是：
# 
# - 5个卷积层，除第4个卷积层外，每个卷积层后面紧跟着1个风格损失层
# - 第4个卷积层紧跟着内容损失层，然后是风格损失层
# - 其它的relu和maxpool跟vgg的前5层一致
# 
# Sequential(
# 
#   (0): Norm()
#   
#   (conv_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   
#   (style_loss_1): StyleLoss()
#   
#   (relu_1): ReLU()
#   
#   (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   
#   (style_loss_2): StyleLoss()
#   
#   (relu_2): ReLU()
#   
#   (pool_2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   
#   (conv_3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   
#   (style_loss_3): StyleLoss()
#   
#   (relu_3): ReLU()
#   
#   (conv_4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   
#   (content_loss_4): ContentLoss()
#   
#   (style_loss_4): StyleLoss()
#   
#   (relu_4): ReLU()
#   
#   (pool_4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   
#   (conv_5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   
#   (style_loss_5): StyleLoss()
# )

#%%
img_input=img_content.clone()
img_show(img_input,title='Input Image')

#%% [markdown]
# # 梯度下降
# 
# 如同算法的作者Leon Gatys说过那样（https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq） ，我们将使用L-BFGS算法来执行梯度下降.
# 
# 和训练一个网络不一样，我们训练输入图像，是为了最小化内容和风格损失.
# 
# 通过使用optim.LBFGS来创建一个L-BFGS优化器，然后把图像张量传给它来做优化.

#%%
##优化的目标是输入图像本身,不再是模型的参数
def get_input_optimizer(input_img):
    opt=optim.LBFGS([input_img.requires_grad_()])##再次注意，是对输入求导
    return opt

#%% [markdown]
# 最后，必须定义一个函数来执行神经转移. 对于网络的每次迭代来说，每次提供一个更新的输入，并计算新的损失.
# 
# 我们会运行每个损失模块的.backward方法，来动态计算它们的梯度.
# 
# 优化器需要一个闭包函数，用作对模型的重评估并返回损失.
# 
# 仍然有最后一个限制需要解决，就是网络可能优化的输入值将会超出[0,1]之间的范围.
# 
# 将通过纠正输入值来解决该问题.

#%%
def run_style_transfer(cnn,mean,std,img_content,img_style,img_input,num_steps=300,style_weight=1000000,content_weight=1):
    print('building style transfer model...')
    
    model,style_losses,content_losses=get_style_module_and_losses(cnn,mean,std,img_style,img_content)
    ## 返回值:
    ## model ：对vgg19进行过拆解并添加了内容和风格损失层的模型
    ## style_losses:  风格损失层：包括风格的目标特征图，以及计算输入特征图和目标特征图之间损失的forward函数
    ## content_losses:内容损失层：包括内容的目标特征图，以及计算输入特征图和内容特征图之间损失的forward函数
    
    
    
    opt=get_input_optimizer(img_input)
    
    print('optimizing...')
    run=[0]
    while run[0]<=num_steps:
        ##迭代中定义的closure，供LBFGS优化器使用
        def closure():
            ##纠偏
            img_input.data.clamp_(0,1)
            
            opt.zero_grad()
            ##对输入图像进行前向传递
            ##同时计算输入特征图和内容特征图之间的损失，以及输入特征图和风格特征图之间的损失
            model(img_input)
            style_score=0
            content_score=0
            
            ##因为计算过的损失，已经存储在风格层或损失层的loss属性中，
            ##所以这里取出
            for sl in style_losses:
                style_score+=sl.loss
            for cl in content_losses:
                content_score+=cl.loss
            
            ##再对损失进行加权，有所侧重，
            ##这里style权重默认较大，为1w
            style_score *= style_weight
            content_score *= content_weight
            
            ##最终的目标损失是：输入与风格损失+输入与内容损失的加权和.
            loss=style_score+content_score
            ##反向传播，计算关于输入图像的梯度
            loss.backward()
            
            run[0]+=1
            if run[0]%50==0:
                print('run {}:'.format(run))
               # print('Style Loss :{:.4f} Content Loss:{:.4f}'.format(style_score.item(),content_score.item()))
                print()
                
            return style_score+content_score
        
        ##LBFGS的优化步中需要closure作为参数
        opt.step(closure)
        
    ##对数据进行纠偏    
    img_input.data.clamp_(0,1)
    return img_input


#%%
output=run_style_transfer(model,mean,std,img_content,img_style,img_input)
img_show(output,'Output Image')


#%%
print('*'*100)


#%%
vgg=tv.models.vgg19(pretrained=True)


#%%
vgg.features


#%%
i=0
for m in vgg.features.children():
    if isinstance(m,nn.Conv2d):
        i+=1


#%%
print(i)


#%%
len(vgg.features)


#%%
for i in range (5,-1,-1):
    print(i)


#%%
i


#%%
m,style_losses,content_losses=get_style_module_and_losses(model,mean,std,img_style,img_content)


#%%
m


#%%
content_losses


#%%
style_losses


#%%



