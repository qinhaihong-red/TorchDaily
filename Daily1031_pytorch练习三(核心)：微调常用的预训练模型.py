
#%%
import os,time,copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 对torchvision的模型进行微调
# 
# 本篇研究一下如何对torchvision的模型进行微调和特征提取，而这些模型全部在Imagenet的1000分类数据集上进行过预训练.
# 
# 这里我们将会进行两种形式的迁移学习：
# 
# - finetuning 微调：从一个已经预训练的模型开始，然后对模型的所有参数进行更新，本质上就是把模型从头再训练一次.
# 
# 
# - feature extraction 特征提取：只更新最后的全连接层，因为该层给出了我们想要的预测结果. 称之为特征提取的原因，是因为保持了模型的所有参数不变，只更换了全连接层.
# 
# 更多关于迁移学习的技术信息，参考：
# 
# http://cs231n.github.io/transfer-learning/
# 
# http://ruder.io/transfer-learning/
# 
# 
# 
# 
# 大体来说，迁移学习遵循以下步骤：
# 
# - 初始化预训练模型
# 
# 
# - 更换其全连接层，并使之输出与当前训练数据集的分类一致
# 
# 
# - 定义损失函数和优化算法
# 
# 
# - 执行训练步
#%% [markdown]
# # 1. 定义相关的输入变量 

#%%
data_dir='./hymenoptera_data/'

batch_size=4
num_classes=2
num_epochs=10

#%% [markdown]
# # 2.定义模型训练函数和帮助函数
#%% [markdown]
# ## 2.1 训练模型函数
#%% [markdown]
# - 该函数的输入为model,dataloader,E,optimizer,num_epochs,is_inceptionv3
# 
# - 由于inceptionv3模型需要辅助输出，以及针对辅助输出和最终输出的模型的所有损失，需要is_inceptionv3来做适配. 见：https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
# 
# - 函数返回最优模型以及输出最佳Acc. 基本上与上一篇的train函数类似.

#%%
def train_model(model,dataloader,E,opt,lr_scheduler,num_epochs,is_inceptionv3):
    start = time.time()
    best_acc = 0.0
    best_model_params=copy.deepcopy(model.state_dict())
    
    for i in range(num_epochs):
        print('epoch:{}/{}'.format(i,num_epochs-1))
        print('-'*10)
        
        
        for phase in ['train','val']:
            if phase == 'train':
                lr_scheduler.step()
                model.train()
            else:
                model.eval()
                
            running_loss=0.0
            running_corrects=0.0
            
            for j,(inputs,labels) in enumerate(dataloader[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):                    
                    
                    if phase == 'train' and is_inceptionv3 is True:
                        out,out2=model(inputs)
                        loss1=E(out,labels)
                        loss2=E(out2,labels)
                        loss=loss1+0.4*loss2
                    
                    else:
                        out=model(inputs)
                        loss=E(out,labels)
                        
                    _,preds=torch.max(out,dim=1)   
                    
                    if phase == 'train':    
                        loss.backward()
                        opt.step()
                        
                    running_loss=running_loss+loss.item()*inputs.shape[0]
                    running_corrects = running_corrects+torch.sum(labels.data==preds)
                    
            epoch_loss=running_loss/(len(dataloader[phase].dataset))
            epoch_acc=running_corrects.double()/(len(dataloader[phase].dataset))
            
            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_params = copy.deepcopy(model.state_dict())
        
        print()
        
    time_eplase = time.time()-start
    print('Train completed in {:.0f}m {:.0f}s'.format(time_eplase//60,time_eplase%60))
    print('best Acc:{:.4f}'.format(best_acc))

    model.load_state_dict(best_model_params)

    return model

#%% [markdown]
# ## 2.2 为feature_extract定义关闭求导的函数

#%%
def set_params_req_grad(model,feature_extract):
    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False

#%% [markdown]
# # 3.模型初始化以及重整
# 
# 要使用预训练模型，需要对每个模型进行重整(reshaping)，且依据模型的不同，重整方式也不同.
# 
# 由于所有的模型都是在Imagenet 1000分类数据上进行的训练，因此它们的最后输出都具有1000个结点：每个点对应一个类.
# 
# 这里的重整目标是，对最后一层进行替换，使得该层的输入与模型之前的输出保持一致，并且输出符合新数据集的分类要求.
# 
# 下面将讨论根据不同的模型，对模型的架构进行调整.
# 
# 首先要看看关于微调 finetuning 与 特征提取feature-extraction 重要细节方面的差异：
# 
# __特征提取:__
# 
# 在此过程中，只更新最后一层的参数，或者说，只更新被重整那层的参数. 
# 
# 因此为了效率，没有必要计算那些不改动参数的梯度，要把这些参数的requires_grad属性置为False. 而在默认情况下，这些参数的requires_grad属性为True. 然后，初始化新的最后那层，在默认情况下，该层的requires_grad属性为True，这是我们需要的.
# 
# 
# __微调:__
# 
# 在做微调的时候，由于需要更新所有的参数，因此默认保持所有参数的requires_grad属性为True即可.
# 
# __注意：__
# 
# inception_v3要求的输入尺寸为(299,299), 而其他所有的模型要求为(224,224)
#%% [markdown]
# ## 3.1 resnet
# 
# resnet见于：https://arxiv.org/abs/1512.03385
# 
# resnet有诸多变种，这里使用resnet18.
# 
# 通过查阅源码或者打印一个resnet模型，可发现它的最后一层是：
# 
# > (fc): Linear(in_features=512, out_features=1000, bias=True)
# 
# 由于我们只有两个分类，因此可做如下替换：
# 
# > model.fc = nn.Linear(512,num_classes)
#%% [markdown]
# ## 3.2 alexnet
# 
# alexnet见于：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# 
# 通过查阅源码或者打印模型，可以发现它的最后一层是：
# 
# > (classifier): Sequential(
# 
# >   ...
# 
# >    (6): Linear(in_features=4096, out_features=1000, bias=True)
# 
# > )
# 
# 
#%% [markdown]
# 做如下替换：
# 
# > model.classifier[6] = nn.Linear(4096,num_classes)
#%% [markdown]
# ## 3.3 vgg
# 
# 见于：https://arxiv.org/pdf/1409.1556.pdf
# 
# vgg有诸多变种，这里使用vgg11.
# 
# 通过查阅源码或者打印模型，可以发现它的最后一层是：
# 
# > (classifier): Sequential(
# 
# >   ...
# 
# >    (6): Linear(in_features=4096, out_features=1000, bias=True)
# 
# > )
# 
# 与alexnet一致.
# 
# 做如下替换：
# 
# > model.classifier[6] = nn.Linear(4096,num_classes)
#%% [markdown]
# ## 3.4 squeezenet
# 
# 见于：https://arxiv.org/abs/1602.07360
# 
# 与其他模型相比，squeezenet使用了一种不同的输出结构.
# 
# Torchvision具备两种不同的squeezenet，即1.0版和1.1版，这里使用1.0版.
# 
# 它的输出层是classifier中的第1层（从0开始），是一个kernel_size为（1，1），输入通道为512，输出通道为1000的卷积层:
# 
# > (classifier): Sequential(
# 
# >   (0): Dropout(p=0.5)
# 
# >    (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
# 
# >    (2): ReLU(inplace)
# 
# >    (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
# > )
# 
# 
# 替换为：
# 
# > model.classifier[1] = nn.Conv2d(512,num_classes,kernel_size=(1, 1), stride=(1, 1))
#%% [markdown]
# ## 3.5 densenet
# 
# 见于：https://arxiv.org/abs/1608.06993
# 
# 有诸多变种，这里使用densenet-121.
# 
# 它的输出层是个输入为1024的线性层：
# 
# > (classifier): Linear(in_features=1024, out_features=1000, bias=True)
# 
# 替换为：
# 
# > model.classifier = nn.Linear(1024,num_classes)
#%% [markdown]
# ## 3.6 inceptionv3
# 
# 见于：https://arxiv.org/pdf/1512.00567v1.pdf
# 
# 这个网络也堪称独一无二的, 因为它在训练阶段具备两个输出层. 其中第二个输出层作为辅助输出，包含在网络的AuxLogits部分.
# 
# 主要的输出是位于网络末端的线性层. 需要注意的是，在测试中，我们主要使用首要输出.
# 
# 辅助输出和首要输出的描述如下：
# 
# > (AuxLogits): InceptionAux(
# 
# >    ...
# 
# >    (fc): Linear(in_features=768, out_features=1000, bias=True)
# 
# > )
# 
# > ...
# 
# > (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 
# 必须同时对这两个层进行重整：
# 
# > model.AuxLogits.fc = nn.Linear(768,num_classes)
# 
# > model.fc = nn.Linear(2048,num_classes)
#%% [markdown]
# ## 3.7 定义初始化模型的函数
# 
# 每个模型都具备类似的输出结构，但是又有细微的差别.
# 
# 必要时打印出模型参数或者查看源码，以确认替换过的输出层，既符合原来的输入要求，又符合新的数据集的分类要求.

#%%
def init_model(model_name,num_classes,feature_extraction,pretrained=True):
    model = None
    intpu_size= 0 
    
    if model_name == 'resnet':
        model = tv.models.resnet18(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        in_features=model.fc.in_features
        model.fc=nn.Linear(in_features,num_classes)
        input_size=224
    elif model_name == 'alexnet':
        model = tv.models.alexnet(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        in_features=model.classifier[6].in_features
        model.classifier[6].in_features=in_features
        input_size=224
    elif model_name == 'vgg':
        model = tv.models.vgg11(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        in_features=model.classifier[6].in_features
        model.classifier[6].in_features=in_features
        input_size=224
    elif model_name =='squeezenet':
        ##输出层是一个卷积层，位于classifier下标为1的位置
        ##同时要更新num_classes属性
        model = tv.models.squeezenet1_0(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        in_channels=model.classifier[1].in_channels
        kernel_size=model.classifier[1].kernel_size
        new_outlayer = nn.Conv2d(in_channels,num_classes,kernel_size)
        model.classifier[1]=new_outlayer
        model.num_classes=num_classes
        input_size=224
    elif model_name == 'densenet':
        model = tv.models.densenet121(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        in_features=model.classifier.in_features
        model.classifier=nn.Linear(in_features,num_classes)
        intput_size=224
    elif model_name == 'inception':
        model = tv.models.inception_v3(pretrained=pretrained)
        set_params_req_grad(model,feature_extraction)
        #辅助输出
        in_features=model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features,num_classes)
        ##主要输出
        in_features=model.fc.in_features
        model.fc = nn.Linear(in_features,num_classes)
        input_size=299
        
    
    return model,input_size

#%% [markdown]
# # 4.数据加载
# 
# ## 4.1 数据增强与变换
# 

#%%
def get_data_transforms(input_size):

    data_transforms={
        'train':tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(input_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ]),
        'val':tv.transforms.Compose([
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])
    }
    
    return data_transforms

#%% [markdown]
# ## 4.2构造数据集和加载器
# 

#%%
img_datasets={x:tv.datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
img_dataloader={x:torch.utils.data.DataLoader(img_datasets[x],4,shuffle=True) for x in ['train','val']}

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% [markdown]
# # 5.构造优化器
#%% [markdown]
# 根据对模型微调方式的不同，需要学习的参数也不同，我们希望只优化那些需要学习的参数.
# 
# - finetuning: 所有参数
# 
# - feature extraction: 替换层的参数

#%%
input_size


#%%
feature_extract=False
model_name='inception'
is_inception = (model_name == 'inception')

model,input_size=init_model(model_name,2,True)
data_transforms = get_data_transforms(input_size)

#model=tv.models.resnet18(pretrained=True)
#in_features=model.fc.in_features
#model.fc=nn.Linear(in_features,num_classes)

model=model.to(device)#send model to gpu

params_to_learn = model.parameters()
if feature_extract:
    params_to_learn = []
    
    #如果只使用resnet，可以只把model.fc传给优化器即可
    ##但是如果使用其他模型，替换的层都不一样，因此需要逐一检测requires_grad，收集需要学习的参数
    for p in model.parameters():
        if p.requires_grad:
            params_to_learn.append(p)
        
        
opt=optim.SGD(params_to_learn,lr=0.01,momentum=0.9)
E=nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(opt,step_size=2,gamma=0.1)


#%%
model=train_model(model,img_dataloader,E,opt,lr_scheduler,num_epochs,is_inception)

#%% [markdown]
# # 6. 从头开始训练的模型和使用预训练的模型，在验证集上的对比
# 
# 通常来说，通过迁移学习方法利用预训练的模型，比从头训练的模型，无论从时间消耗和准确率来说，都有优势.
# 
# 可以看到从头训练的模型，其正确率，达不到50%.
# 
# ![image](https://pytorch.org/tutorials/_images/sphx_glr_finetuning_torchvision_models_tutorial_001.png)
# 
# 参考：https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#comparison-with-model-trained-from-scratch

#%%



