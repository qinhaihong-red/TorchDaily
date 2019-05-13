
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#%% [markdown]
# # 1.优化器包 torch.optim
# 
# torch.optim包，实现了多种优化算法. 
# 
# 该包支持绝大多数优化算法，并且接口扩展性好，将来更多复杂的算法可以轻松集成到包中.
# 
#%% [markdown]
# # 2. 如何使用一个优化器
# 
# 为使用优化器，必须先构建一个优化器对象.
# 
# 该对象持有当前待优化参数的状态，并根据已计算的梯度，对参数进行更新.
# 
# ## 2.1 构造优化器
# 
# 为构造优化器，必须传给它一个可迭代的包含待优化参数的变量. 然后可以为优化器指定具体的优化选项，如学习率、权重衰减等.
# 
# __注意：__
# > 如果既需要为模型的参数构造优化器，又要通过.cuda方法，把模型移动到GPU上，那么应该先构造优化器再移动到GPU。
# > 这是因为模型的参数在调用.cuda前后，已经是不同的对象了.
# 
# > 通常来说，当优化器构造并使用参数时，需要保证参数变量位于稳定的区域.
#%% [markdown]
# 示例：

#%%
mod=nn.Linear(2,3)
mod.parameters()


#%%
for n,i in mod.named_parameters():
    print(n,'-->',i)


#%%
opt=optim.SGD(mod.parameters(),lr=0.01,momentum=0.9)


#%%
##注意优化器的param_groups格式
opt.param_groups


#%%
x1=torch.randn(2,3)
x2=torch.randn(3)
opt2=optim.Adam([x1,x2],lr=0.0001)
opt2.param_groups

#%% [markdown]
# ## 2.2 为每个参数指定选项
# 
# 优化器支持为每个参数都设置具体的选项. 要想实现这个目标，需要传给优化器构造函数一个可迭代字典.
# 
# 需要为字典中每个对象定义一个参数组：即包括一个key为params，value为待优化的参数以及为该参数指定的选项.
# 
# 字典外的参数，被视为对字典内所有对象都使用的优化参数.
#%% [markdown]
# 示例：

#%%
from collections import OrderedDict
mod=nn.Sequential(OrderedDict({
    'conv1':nn.Conv2d(3,3,3,padding=1),
    'relu1':nn.ReLU(),
    'fc':nn.Linear(2,1)
}))


#%%
mod._modules['conv1'].parameters

#%% [markdown]
# 当为不同层的参数指定不同优化策略时，使用这种方法就很有用.
# 
# 外层的lr和momentum对所有参数都适用，如果内层定义了自己的参数，则不会受到外层影响.

#%%
opt=optim.SGD(
    [
        {'params':mod._modules['conv1'].parameters()},
    
        {'params':mod._modules['fc'].parameters(),'lr':1e-4}    
    ],lr=1e-2,momentum=0.9)


#%%
opt.param_groups

#%% [markdown]
# ## 2.3 采取一步优化 Taking an optimization step
# 
# 所有优化器都实现了step()方法，通过该方法对待优化参数进行更新.
# 
# 有两种使用step()的方式：
# 
# ### 2.31 不带closure的step()方法：
# 
# 每次当计算完待优化参数后，即调用backward()之后，即可调用step().
# 
# 使用示例如下：

#%%
##1.以mod的参数构造opt
##2.
for input,target in dataset:
    opt.zero_grad()
    out=mod(input)
    loss=loss_fn(out,target)
    loss.backward()
    opt.step()

#%% [markdown]
# ### 2.32 带参数的step(closure)方法：
# 
# 有些优化算法，如共轭梯度和LBFGS需要多次对函数进行再评估. 
# 
# 因此要给step传入一个closure，允许其对模型再次计算.
# 
# 闭包应该清空梯度、计算损失并返回.
# 
# 示例：

#%%
for input,target in dataset:
    def closure():
        opt.zero_grad()
        out=model(input)
        loss=loss_fn(out,target)
        return loss
    
    opt.step(closure)

#%% [markdown]
# # 3.具体优化器算法
# 
# ## 3.1 优化器基类 Optimizer(params, defaults)
# 
# __注意:__
# 
# >传入优化器构造函数的可迭代集合，应该是在优化期间具有稳定排序的集合.
# 
# >不满足该性质的集合之一为sets。
# 
# 
#%% [markdown]
# 方法：
# 
# - param_groups
# 
# 返回所有参数组. 见2.2
# 
# - add_param_group(param_group)
# 
# 为优化器添加一个参数组.
# 
# 当微调一个预训练的模型时，该方法比较有用.
# 
# 可以把某层的参数用次方法传给优化器，实现只对某层参数进行优化.
#%% [markdown]
# - load_state_dict(state_dict)
# 
# 加载优化器状态.
# 
# 参数应该是state_dict函数的返回值.
#%% [markdown]
# - state_dict()
# 
# 以字典形式返回当前优化器状态.
# 
# 包括两个入口：
#     - state: 持有当前优化器状态的字典. 不同优化器对象之间内容不同.
#     - param_groups: 一个包含所有参数组的字典.
#%% [markdown]
# - step()
# 
# 略.
#%% [markdown]
# - zero_grad()
# 
# 清空所有待优化张量的梯度.
#%% [markdown]
# ## 3.2 Adadelta算法
# 
# - optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# 
# 论文见：https://arxiv.org/abs/1212.5701
#%% [markdown]
# ## 3.2 Adagrad 算法
# 
# - optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
# 
# 论文见：http://jmlr.org/papers/v12/duchi11a.html
#%% [markdown]
# ## 3.3 Adam 算法
# 
# - optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 
# 论文见：https://arxiv.org/abs/1412.6980
#%% [markdown]
# ## 3.4 SparseAdam
# 
# 适用于稀疏张量的懒惰版本的Adam算法.
# 
# - torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
#%% [markdown]
# ## 3.5 Adamax
# 
# 基于Adam使用无穷范数的Adam算法变种.
# 
# - optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# 
# 论文见：https://arxiv.org/abs/1412.6980
#%% [markdown]
# ## 3.6 ASGD
# 
# 平均SGD算法.
# 
# -  optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# 
# 论文见：http://dl.acm.org/citation.cfm?id=131098
#%% [markdown]
# ## 3.7 LBFGS
# 
# - optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
#%% [markdown]
# ## 3.8 RMSprop
# 
# - optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# 
# 见：
# 
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# 
# https://arxiv.org/pdf/1308.0850v5.pdf
#%% [markdown]
# ## 3.9 Rprop
# 
# 弹性反向传播算法.
# 
# - optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
#%% [markdown]
# ## 3.10 SGD
# 
# - optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# 
# SGD算法，动量参数可选.
# 
# Nestero动量基于的方程源自：http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
# 
# 
#%% [markdown]
# __注意：__
# 
# >带有动量/nesterov的SGD算法，与Sutskerver等人以及其他框架所实现的SGD算法，有微妙的区别.
# 
# 考虑带动量的情况，更新过程如下：
# 
# $v_{t}=\rho v_{t-1}+ g_{t}$
# 
# $p_{t}=p_{t-1}-lr*v_{t}$
# 
# 其中，$p表示参数，\rho 表示动量，g表示梯度，v表示速率$.
# 
# 其他框架的实现如下：
# 
# $v_{t}=\rho * v_{t-1} + lr * g_{t}$
# 
# $p_{t}=p_{t-1}-v_{t}$
# 
# 
# 带Nesterov参数的版本也有类似的修改.
#%% [markdown]
# # 4. 如何调整学习率
# 
# torch.optim.lr_scheduler包提供了基于epochs数目，调整学习率的一些方法.
# 
# torch.optim.lr_scheduler.ReduceLROnPlateau 允许基于验证测量的动态学习率减小方法.
#%% [markdown]
# ## 4.1 LambdaLR 单步控制
# 
# - optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
# 
# 为每个参数组设置学习率：即初始化的学习率与给定的lambda函数相乘.
# 
# 这个lambda函数通常是关于epochs的函数，也就是根据epochs，来调整学习率.
# 
# 示例：

#%%
lam1=lambda epoch:epoch // 30
lam2=lambda epoch:0.95 * epoch

##参数组1根据lam1调整学习率
##参数组2根据lam2调整学习率
scheduler=optim.lr_scheduler.LambdaLR(opt,lr_lambda=[lam1,lam2])
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)

#%% [markdown]
# ## 4.2 MultiStepLR 多步控制
# 
# - optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
# 
# 为每个参数组设置里程碑epoch，当参数组达到这个里程碑时，将通过gamma参数对先前的lr进行衰减，对lr进行控制.
# 
# 示例：
# 
# 假设每个参数组使用相同的初始化lr=0.001
# 
# lr = 0.001 , if epoch < 30
# 
# lr = 0.001 * gamma, if epoch <=30 <80
# 
# lr = 0.001 \* gamma \* gamma , if epoch >= 80

#%%
scheduler = optim.lr_scheduler.MultiStepLR(opt,milestones=[30,80],gamma=0.001)
for epoch in range(100):
    scheduler.step()
    train()...
    val()...

#%% [markdown]
# __当达到最后一个epoch（默认-1代表最后一个）时，会把lr再设置为初始lr.__
#%% [markdown]
# ## 4.3 ExponentialLR 指数控制
# 
# - optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
# 
# 不再设定一个或多个epoch，即每个epoch都对lr进行gamma衰减，因此是一种指数的衰减.

#%%
mod


#%%
opt.param_groups


#%%
lr=optim.lr_scheduler.ExponentialLR(opt,0.5)
for i in range(10):
    lr.step()


#%%
##param_groups中新增了参数
##initila_lr，是初始化参数
##lr是最后调节的参数
opt.param_groups

#%% [markdown]
# ## 4.4 CosineAnnealingLR 余弦退火
# 
# - torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
# 
# 见：
# 
# https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR
#%% [markdown]
# ## 4.5 ReduceLROnPlateau 平稳减少
# 
# - optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# 
# 当度量指标停止变化时，减少学习率. 一旦学习停滞，通过减少学习率，模型可以从中受益.
# 
# lr调度器会关注指标量，如果几个epochs内没有变化，学习率就会减少.

#%%
for epoch in range(10):
    train()...
    val_loss=validate()
    ##注意这个调度器的调用时机，要在指标量计算完后并传入指标量
    scheduler.step(val_loss)


#%%



