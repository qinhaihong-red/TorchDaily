
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils

#%% [markdown]
# # 1. 瓶颈检测器工具 torch.utils.bottleneck
# 
# torch.utils.bottleneck工具，可以用来对程序的瓶颈进行调试，它综合使用Python的分析器和Pytorch的autograd分析器，
# 对程序进行分析。
# 
# 在命令行中如下运行：
# python -m torch.utils.bottleneck /path/to/source/script.py [args]
# 
# 
# 其中args时script.py所需的参数.
# 
# 更多注意事项见：https://pytorch.org/docs/stable/bottleneck.html
#%% [markdown]
# # 2. 模型存档点 torch.utils.checkpoint
# 
# ## 2.1 checkpoint
# - utils.checkpoint.checkpoint(function, *args)
# 
# 对模型或部分模型进行存档.
# 
# checkingpoint的工作方式为以算力换内存. 与其存储在反向过程中产生的属于计算图的所有中间激活，checkpoint是在反向传播过程中对它们再次进行计算. 这种方式可以在模型的任何部分使用.
# 
# 尤其需要指出，在前向传递中，function以torch.no_grad方式运行，即并不存储所有的中间激活. 同时，前向传递会保存输入元组和function参数.
# 
# 在反向传递中，被保存的输入和函数参数被取，前向传递在function上再次进行，并记录中间激活的产生，然后使用这些激活值计算梯度.
# 
# __注意：__
# > checkpointing只与torch.autograd.backward()匹配，与torch.auto.grad并不匹配.
# 
# > 如果function在反向传递中的表现，与在前向传递中的表现有所不同，被存档的版本将不相同，且无法发现
#%% [markdown]
# ## 2.2 checkpoint_sequential
# 
# - utils.checkpoint.checkpoint_sequential(functions, segments, *inputs)
# 
# 为存档sequential模型准备的帮助函数.
# 
# 顺序模型以一定顺序执行一些模块或函数.因此我们可以把该模型划分为一些段，然后分段存档.
# 
# 除最后一个分段外的所有段，皆以torch.no_grad()方式运行，即并不存储中间激活. 每个存档段的输入参数将会被保存，以待在反向传递中再次运行
# 该段时使用.
# 
# 示例：

#%%
mod=nn.Sequential(...)
input_var=utils.checkpoint.checkpoint_sequential(...)


