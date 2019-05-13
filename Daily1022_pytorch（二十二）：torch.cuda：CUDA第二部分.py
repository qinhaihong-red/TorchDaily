
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda

#%% [markdown]
# # 1. cuda包：torch.cuda
# 
# 此包为CUDA张量类型增加了支持，即实现了与CPU张量一样的函数，只是为算力而利用了GPU.
# 
# 这个包总是懒初始化的，因此总是可以引入它. 通过使用is_available来判断系统是否支持CUDA.
# 
# 阅读 CUDA语义 了解更多细节：https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
#%% [markdown]
# - torch.cuda.current_blas_handle()
#%% [markdown]
# - torch.cuda.current_device():返回当前所选设备的索引

#%%
index=cuda.current_device()
index

#%% [markdown]
# - torch.cuda.device_count():返回当前可用的GPU数目

#%%
num=cuda.device_count()
num

#%% [markdown]
# - torch.cuda.get_device_capability(device): 返回设备的cuda能力

#%%
cuda.get_device_capability(0)

#%% [markdown]
# - torch.cuda.get_device_name(device):返回设备名称

#%%
cuda.get_device_name(0)

#%% [markdown]
# - torch.cuda.max_memory_allocated(device):返回指定设备张量的最大GPU内存用量

#%%
cuda.max_memory_allocated(0)


#%%
device=torch.device('cuda') if cuda.is_available() else torch.device('cpu')
X=torch.randn(100,100,device=device)
X.shape


#%%
cuda.max_memory_allocated(0)

#%% [markdown]
# - torch.cuda.max_memory_cached(device=None):返回指定设备缓存分配器管理的最大GPU内存

#%%
cuda.max_memory_cached(0)

#%% [markdown]
# - torch.cuda.memory_allocated(device=None):返回指定设备上张量使用的GPU内存

#%%
cuda.memory_allocated(0)

#%% [markdown]
# - torch.cuda.memory_cached(device=None)

#%%
cuda.memory_cached()

#%% [markdown]
# -  orch.cuda.set_device(device):设置当前的设备. 该函数不鼓励使用. 更好的方法为使用CUDA_VISIBLE_DEVICES.
#%% [markdown]
# - torch.cuda.stream(stream):用于选定流的上下文管理器.
#%% [markdown]
# - torch.cuda.synchronize():在当前设备上等待所有流中核操作的结束.
#%% [markdown]
# # 2.随机数生成器
#%% [markdown]
# - torch.cuda.get_rng_state(device=-1):返回当前GPU的随机数生成器状态.
#%% [markdown]
# - torch.cuda.set_rng_state(new_state, device=-1):设置当前GPU的随机数生成器状态.
#%% [markdown]
# - torch.cuda.manual_seed(seed):为当前GPU设置生成随机数的种子
#%% [markdown]
# - torch.cuda.manual_seed_all(seed):为多GPU设置生成随机数的种子.
#%% [markdown]
# - torch.cuda.seed():为当前GPU设置随机数种子.
#%% [markdown]
# - torch.cuda.seed_all():为多GPU设置随机数种子.
#%% [markdown]
# - torch.cuda.initial_seed():返回当前GPU的种子.
#%% [markdown]
# # 3. 通信集体 Communication Collectives
#%% [markdown]
# - torch.cuda.comm.broadcast(tensor, devices):把一个张量广播到给定数量的GPU上.
#%% [markdown]
# - torch.cuda.comm.broadcast_coalesced(tensors, devices, buffer_size=10485760):把指定的张量序列广播到指定设备上
#%% [markdown]
# - torch.cuda.comm.reduce_add(inputs, destination=None):对多GPU上的张量求和.
#%% [markdown]
# - torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None): 把张量分散到多GPU上.
#%% [markdown]
# - torch.cuda.comm.gather(tensors, dim=0, destination=None):把多GPU上的张量进行聚集.
#%% [markdown]
# # 4.流和事件 Streams and Events
#%% [markdown]
# ## 4.1 class torch.cuda.Stream
# 
# 对一个CUDA stream的封装.
# 
# stream是属于特定设备的、由线性可执行操作组成的序列，与其他的流独立.
# 
# 类似一个支持并发的队列.
# 
# __方法：__
# 
# - query:所有提交的任务是否完成
# 
# 
# - record_event:
# 
# 
# - synchronize:
# 
# 
# - wait_event:
# 
# 
# - wait_stream:
#%% [markdown]
# ## 4.2 class torch.cuda.Event
# 
# 对CUDA事件的封装类.
# 
# - elapsed_time:
# 
# 
# - ipc_handle:
# 
# 
# - query:
# 
# 
# - record:
# 
# 
# - synchronize:
# 
# 
# - wait:
#%% [markdown]
# # 5. 内存管理
# 
# - torch.cuda.empty_cache():释放所有当前被缓存分配器持有的未被使用的缓存内存，使得这些被释放的内存可被其他GPU应用程序使用，并在
# nvidia-smi中被发现.
# 
# 注意这个函数不会增加Pytorch的GPU内存可用量. 见：https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
#%% [markdown]
# - 其他memory_allocated和memory_cached见上述，不再赘述.
#%% [markdown]
# # 6. nvidia 工具扩展（NVTX）
#%% [markdown]
# - torch.cuda.nvtx.mark(msg)
# 
# 
# - torch.cuda.nvtx.range_push(msg)
# 
# 
# - torch.cuda.nvtx.range_pop()

#%%



