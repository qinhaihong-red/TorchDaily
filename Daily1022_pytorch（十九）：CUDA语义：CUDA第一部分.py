
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tf
import torchvision.models as mod
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # CUDA语义
# 
# - 通过使用torch.cuda来创建和运行CUDA操作. 它会追踪当前所选的GPU，所有分配的CUDA张量默认都会创建在这个GPU设备上.
# 
# 
# - 可以通过torch.cuda.device上下文管理器改变当前所选的设备.
# 
# 
# - 一旦张量被分配，可以在上面做任何操作，而不用考虑所选的设备，并且所有的结果都会像之前被操作的张量一样，置于同样的设备上.
# 
# 
# - 跨GPU操作默认是不被允许的，除了copy\_()方法以及类似的函数，如to()和cuda()函数.
# 
# 
# - 除非允许点到点的内存存取，否则在任何跨设备的张量上进行操作都会引发错误.
# 
# CUDA示例如下：
#%% [markdown]
# ### 创建设备

#%%
cuda=torch.device('cuda')
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu=torch.device('cpu')


#%%
cuda


#%%
device


#%%
cpu


#%%
cuda0=torch.device('cuda:0')
cuda0


#%%
##这些超过0的GPU序数，尽管可以定义，但是不能使用
##因为只有一个GPU

cuda1=torch.device('cuda:1')
cuda1


#%%
cuda2=torch.device('cuda:2')
cuda2

#%% [markdown]
# ### 分配张量

#%%
x=torch.arange(3)
y=torch.arange(3,6,device='cuda')
z=torch.arange(7,10).cuda()


#%%
x


#%%
x2=torch.full((3,),fill_value=1,dtype=torch.long)
x2


#%%
y


#%%
z

#%% [markdown]
# ### 操作

#%%
##两个cpu上的张量相加
ret1=x+x2
ret1


#%%
##类型不匹配，导致cpu上的tensor与gpu上的tensor无法直接相加
ret2=x+y
ret2


#%%
##两个GPU上的张量相加
##结果仍然为GPU上的张量
ret3=y+z
ret3

#%% [markdown]
# ###  互转

#%%
y1_cpu=torch.randn(2)
y1_cpu


#%%
y1_gpu=y1_cpu.to(device)
y1_gpu


#%%
y1_gpu=y1_cpu.cuda('cuda')
y1_gpu


#%%
y1_cpu=y1_gpu.to('cpu')
y1_cpu

#%% [markdown]
# ## 1. 异步执行
# 
# GPU操作默认是异步执行的。当调用一个使用GPU的函数时，该操作被入队于特定设备，但是并非直到之后才执行. 这允许我们可以并行地执行更多计算，
# 包括CPU和其他GPU上的操作.
# 
# 一般来说，异步调用的效果对调用者是不可见的，因为：
# 
# - 每个入队设备的操作，以入队顺序被设备执行
# 
# - 当在CPU和GPU之间或者GPU与GPU之间进行拷贝数据时，Pytorch会自动执行必备的同步操作
# 
# 因此计算将会持续进行，犹如每个操作都是同步执行的一样.
# 
# 通过设置环境变量 CUDA_LAUNCH_BLOCKING=1 ，可以强制使用同步计算. 当在GPU上发生错误时，这个操作可能很有用.
# (因为当异步执行时，直到操作被执行后发生错误时，才会报告错误，此时堆栈追踪并不会显示是哪里导致的错误)
# 
# 作为一个例外，有些函数，如copy\_(), 会使用额外的aync参数，能让调用者在不必要的时候绕过同步操作。另个例外是CUDA流（streams），如下所示。
#%% [markdown]
# ## 2.CUDA流
# 
# 一个CUDA stream是属于一个特定设备的、由线性操作操作构成的序列.
# 
# 一般没有必要显示创建一个：每个设备有一个默认的流.
# 
# 每个流内部的操作以它们被创建的顺序进行序列化，但是不同流内部的不同操作，能以任何相对顺序并发执行，除非使用显式的同步函数（如synchronize或wait_stream）
# 
# 
# 在当前流是默认流的情况下，Pytorch会在数据进行移动时，自动执行必备的同步操作. 而如果使用自定义流，那么用户需要自己来保证正确的同步操作.
#%% [markdown]
# ## 3. 内存管理
# 
# Pytorch使用一个缓存内存的分配器来加速内存分配操作. 这允许快速进行内存回收而不必借助设备同步. 
# 
# 然而，如果使用nvidia-smi，由分配器管理的未使用的内存，仍然会显示在其中.
# 
# 可以使用memory_allocated和max_memory_allocated来监视被张量占用的内存，同时使用memory_cached和max_memory_cached来监视
# 被缓存分配器管理的内存.
# 
# 调用empty_cache可以释放Pytorch中未使用的缓存内存，这样这些内存可被其他GPU应用使用.
# 
# 但是被张量占用的内存将不会被释放.
#%% [markdown]
# ## 4. 最佳实践
# 
# ### 4.1 使用设备无关的代码
# 
# 
# 主要时通过检测是否有可用的GPU，通过调节判断，给出一个device，如：b
# 

#%%
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


#%%
##CUDA_VISIBLE_DEVICES


#%%
##通过使用new_xx方法，可以创建使用同类型设备的张量
x_cpu=torch.randn(2)
y_gpu=torch.randn(2,device=device)


#%%
y_gpu2=y_gpu.new_full((2,3),fill_value=2)
y_gpu2


#%%
##以及zeros_like或ones_like
y_gpu3=torch.ones_like(y_gpu2)
y_gpu3

#%% [markdown]
# ### 4.2 使用钉住的内存(pinned memory)缓存
# 
# 如果数据源于钉住的内存（锁页内存），那么从主机到GPU的拷贝将会快得多.
# 
# CPU张量和存储，暴露了一个pin_memory的方法，该方法返回对象的拷贝，而该拷贝位于一个钉住的区域.
# 
# 一旦钉住一个张量或存储，就可以使用异步的GPU拷贝. 只需要在调用cuda()的时候，额外传入一个non_blocking=True的参数. 
# 可以在重叠数据转移计算中使用.
# 
# 通过给DataLoader的构造函数传入pin_memory=True的参数，可以使得DataLoader返回位于钉住内存区域的批数据.
#%% [markdown]
# ### 4.3 使用nn.DataParallel进行数据并行而非使用多进程 multiprocessing
# 
# 多数涉及到批输入和多GPU的用例，应该默认使用DataParallel来利用更多的GPU。
# 
# 如果在multiprocessing中使用CUDA模型，会有许多重要的警告. 如果未按需要来确切处理数据, 很可能你的程序会遇到不正确或未定义的行为.

#%%



