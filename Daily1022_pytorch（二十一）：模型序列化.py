
#%%
import numpy as np
import torch
import torch.nn as nn

#%% [markdown]
# #  模型序列化与反序列化
# 
# 有两种方式进行模型的序列化：
# 
# ## 方式一（推荐）：只序列化模型的参数

#%%
def init(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(1.0)
        m.weight.data.fill_(2.0)


#%%
model=nn.Linear(2,3)
model.apply(init)


#%%
for n,i in model.named_parameters():
    print(n,'-->',i)


#%%
##序列化
torch.save(model.state_dict(),'./model/linear_mod.pt')


#%%
##反序列化
mod2=nn.Linear(2,3)
mod2.load_state_dict(torch.load('./model/linear_mod.pt'))


#%%
for n,i in model.named_parameters():
    print(n,'-->',i)

#%% [markdown]
# ## 方式二：序列化全部模型

#%%
torch.save(model,'./model/linear_mod_all.pt')


#%%
mod3=torch.load('./model/linear_mod_all.pt')


#%%
for n,i in model.named_parameters():
    print(n,'-->',i)

#%% [markdown]
# >这种方式下序列化数据被绑定到指定的类，并且使用了特定的目录结构.
# 
# >因此当在其他项目中使用时，可能有很多原因造成模型毁坏.

#%%



