#%% [markdown]
# ## 1.使用CUDA进行加速计算
# 
# 把张量的分配与计算，在GPU上进行，以提高计算速度.
# 
# 见：https://pytorch.org/docs/stable/notes/cuda.html#
#%% [markdown]
# ## 2.使用多GPU进行并行计算
# 
# 在模块层面，使用多GPU，通过切分输入数据，实现并行计算.
# 
# 见:
# 
# https://pytorch.org/docs/stable/notes/cuda.html#use-nn-dataparallel-instead-of-multiprocessing
# 
# 
# https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed
# 
# 
# 以及关于数据并行和计算并行的论文：https://arxiv.org/abs/1404.5997
#%% [markdown]
# ## 3.Pytorch的CPU上的多进程
# 
# 见：
# 
# https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-best-practices
# 
# 
# https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing

#%%



