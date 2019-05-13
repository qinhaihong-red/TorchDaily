
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os,math,time,argparse,random
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
data_root='./data/img_cele/'

img_size=64

transforms=tv.transforms.Compose([tv.transforms.Resize(img_size),
                                  tv.transforms.CenterCrop(img_size),
                                  tv.transforms.ToTensor(),
                                 tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                                 ])
dataset=tv.datasets.ImageFolder(data_root,transform=transforms)

dataloader=torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=128)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def test_show_img():
    batch_img=next(iter(dataloader))
    imgs=batch_img[0]
    img=tv.utils.make_grid(imgs,padding=1,normalize=True)
    img=np.transpose(img,(1,2,0))
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off');

#%%    
##test_show_img()


#%%
##生成器：100x1x1 -> 3x64x64：通道数先升后降，尺寸持续翻倍增加
##使用无偏置的转置卷积，正则化NB，ReLU激活
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(100,8*64,4,stride=1,bias=False),##4*4
            nn.BatchNorm2d(8*64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8*64,4*64,4,stride=2,padding=1,bias=False),##8*8
            nn.BatchNorm2d(4*64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*64,2*64,4,stride=2,padding=1,bias=False),##16*16
            nn.BatchNorm2d(2*64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2*64,64,4,stride=2,padding=1,bias=False),##32*32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64,3,4,stride=2,padding=1,bias=False),##64*64
            nn.Tanh()##生成的图像数据是[-1,1]，展示的时候，需要归一化.      
        )
    
    def forward(self,x):
        return self.main(x)
        
        
##判别器：3x64x64 -> 1x1x1
##使用无偏置的卷积，正则化NB，LeackyReLU
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3,64,4,stride=2,padding=1,bias=False),##32*32,
            nn.LeakyReLU(True),

            nn.Conv2d(64,64*2,4,stride=2,padding=1,bias=False),##16*16,
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(True),

            nn.Conv2d(64*2,64*4,4,stride=2,padding=1,bias=False),##8*8,
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(True),

            nn.Conv2d(64*4,64*8,4,stride=2,padding=1,bias=False),##4*4,
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(True),

            nn.Conv2d(64*8,1,4,stride=1,bias=False),##1*1
            nn.Sigmoid() ##这里已经加了激活，后续的损失函数不用再激活
        )

    def forward(self,x):
        return self.main(x)
    
def init_weights(m):
    if isinstance(m,(nn.ConvTranspose2d,nn.Conv2d)):
        m.weight.data.normal_(0.0,0.02)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

def test_nets():
    netD_test=Discriminator()
    netG_test=Generator()
    netD_test.apply(init_weights)
    netG_test.apply(init_weights)
    
    G_input=torch.randn(2,100,1,1)
    G_out=netG_test(G_input)
    
    D_input=torch.randn(2,3,64,64)
    D_out=netD_test(D_input)
    
    print('G:{}\tD:{}'.format(G_out.shape,D_out.shape))
    
##test_nets()



real_label=1
fake_label=0


img_list=[]
def train(epochs=2,device=device,test=False,pretrained=False):
    
    E=nn.BCELoss()
    model_G_path='./models/modelG.pth'
    model_D_path='./models/modelD.pth'
    
    if pretrained is True:
        modelG=Generator()
        modelD=Discriminator()
        modelG.load_state_dict(torch.load(model_G_path))
        modelD.load_state_dict(torch.load(model_D_path))
        modelG=modelG.to(device)
        modelD=modelD.to(device)
    else:
        modelG=Generator().to(device)
        modelD=Discriminator().to(device)

    fix_data=torch.randn(64,100,1,1,device=device)
    
    optG=optim.Adam(modelG.parameters(),lr=0.0002,betas=(0.05,0.999))
    optD=optim.Adam(modelD.parameters(),lr=0.0002,betas=(0.05,0.999))

    for e in range(epochs):
        for i,(input,target) in enumerate(dataloader):
            
            ##1.训练D
            ##1.1真实数据
            optD.zero_grad()
            input=input.to(device)
            label=input.new_ones((input.shape[0],))#标签全1
            
            out=modelD(input)##真实预测概率
            loss_real=E(out,label)
            loss_real.backward()
            
            D_predreal_mean=out.mean().item()
            
            ##1.2 假数据
            torch.fill_(label,0)#标签全0
            ##modelG的输入:
            noise=torch.randn(input.shape[0],100,1,1,device=device)
            ##使用modelG生成假图片
            fake_img=modelG(noise)
            
            ##假数据预测概率
            ##这里的fake_img要detach一下，因为fake_img是modelG的输出，
            ##如果不detach，那么下面的loss_fake反向传播的时候，会传播到modelG并计算modelG中参数的梯度，
            ##这是没有必要的.
            out=modelD(fake_img.detach())
            loss_fake=E(out,label)
            loss_fake.backward()
            
            loss_D=loss_real+loss_fake
            D_predfake_mean1=out.mean().item()
            
            optD.step()
            
            ##2. 训练G
            optG.zero_grad()
            
            torch.fill_(label,1)
            
            ##这里fake_img不能再detach了，需要通过fake_img这个中间变量，计算关于model参数的梯度.
            ##一旦detach，反向传播到fake_img就停止了，fake_img作为modelG的输出，就不能继续下去了.
            out=modelD(fake_img)
            loss_G=E(out,label)
            loss_G.backward()
            
            D_predfake_mean2=out.mean().item()
            optG.step()
            
            if test is True:
                with torch.no_grad():
                    imgs=modelG(fix_data).detach().cpu()
                    img_list.append(tv.utils.make_grid(imgs,padding=2,normalize=True))
                return 'testOK'
            if i % 100 == 0:
                ##输出统计信息
                print('e:[{}/{}] i:[{}/{}] loss_D:{:.4f} loss_G:{:.4f} pred_D:{:.4f} pred_G1:{:.4f} pred_G2:{:.4f}'.format(
                
                e,epochs,i,len(dataloader),loss_D.item(),loss_G.item(),D_predreal_mean,D_predfake_mean1,D_predfake_mean2
                ))
            
            if i % 500 == 0:
                with torch.no_grad():
                    imgs=modelG(fix_data).detach().cpu()
                    img_list.append(tv.utils.make_grid(imgs,padding=2,normalize=True))##这里的normalize把生成的图像
                    ##从[-1,1]，标准化到[0,1].
            ##全部训练完毕，所需时间太久
            ##i达到150次以后，停止
            #if i>1500:
            #   break
    
    torch.save(modelG.state_dict(),model_G_path)
    torch.save(modelD.state_dict(),model_D_path)
    
    
print('开始训练...')
train(epochs=3,pretrained=False)
print('训练结束.')


#%%
from IPython.display import HTML
def show_img(tensor_img):
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(np.transpose(tensor_img,(1,2,0)));
    
show_img(img_list[8])


#%%
model_G_path='./models/modelG.pth'
model_D_path='./models/modelD.pth'

modelG=Generator()
modelG.load_state_dict(torch.load(model_G_path))
modelG=modelG.to(device)


#%%
fix_data=torch.randn(64,100,1,1,device=device)
imgs=modelG(fix_data).detach().cpu()
imgs=tv.utils.make_grid(imgs,padding=2,normalize=True)
show_img(imgs)


#%%



