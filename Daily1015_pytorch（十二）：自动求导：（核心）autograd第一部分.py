
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1.局部 禁止/恢复 求导
# 
# 在某些情况下，比如进行推断时，没有必要进行求导。同时可以节约内存。
# 
# 可使用的方式如下：
# - __上下文管理器方式__：torch.no_grad/torch.enable_grad
# - __函数方式__：set_grad_enabled. 这个函数也可以用作上下文管理器.

#%%
##1.1 torch.no_grad 禁止求导的上下文管理器
x=torch.randn(3,requires_grad=True)
with torch.no_grad():
    y=x*2
    
y.requires_grad


#%%
##对于函数，可以使用装饰器修饰
@torch.no_grad()
def doubler(x):
    return x*2

z=doubler(x)
z.requires_grad


#%%
##1.2 torch.enable_grad 恢复求导的上下文管理器
##这个管理器可以在禁止求导管理器的内部，恢复求导，但是不会影响到外部
with torch.no_grad():
    with torch.enable_grad():
        y=x*2
y.requires_grad


#%%
@torch.enable_grad()
def doubler2(x):
    return x*2

with torch.no_grad():
    z=doubler2(x)
    
z.requires_grad


#%%
##1.3 禁止/恢复求导函数
torch.set_grad_enabled(False)
y=x*2
y.requires_grad


#%%
torch.set_grad_enabled(True)
z=x*2
z.requires_grad


#%%
##也可以用作上下文管理器
with torch.set_grad_enabled(False):
    y=x*2
y.requires_grad


#%%
torch.set_grad_enabled(True)


#%%
##################################
t=torch.randn(2)
print(t,'\n',t.data,'\n',t.grad)


#%%
t2=torch.randn(2,requires_grad=True)
print(t2,'\n',t2.data,'\n',t2.grad)


#%%
y=x*2
z=y.sum()
z.backward()
y.grad

#%% [markdown]
# ## 2.自动求导机制
#%% [markdown]
# ### 2.1 使用requires_grad从反向传播中排除不需计算的子图
# 
# 
# 
# - 如果对操作的任一输入参数需要梯度，那么输出也要求梯度.
# 就是说，当所有的输入参数都不需要求梯度的时候，那么输出也就不需要求梯度了，这时反向计算过程也就不会执行.
# 
# 
# - 使用requires_grad可以从梯度计算中排除具有良好粒度的、不需要计算梯度的子图，这样能提升运行效率.
#%% [markdown]
# ### 2.2 Autograd:逆向自动微分系统
# 
# ####  理论概念：DAG计算图
# - 通过记录已创建的数据以及在这些数据上执行的操作，autograd得到了一张有向无环图（DAG）
# - 这张图的叶子（leves）节点是输入张量，根节点（root）是输出张量，中间层有临时节点和算子节点
# - 通过对这张图从根节点到叶节点的追踪，使用链式法则（chain-rule），可以求得对应叶子节点的梯度
#%% [markdown]
# #### 内部实现：Function和grad_fn
#%% [markdown]
# - autograd在内部通过Function对象来表示DAG计算图. 即通过调用Function对象的apply函数来计算结果，以求得图的值.
# - 在计算前向传递时，autograd同时执行所请求的计算并建立表示计算函数的图，而torch.Tensor的属性grad_fn则是这张图的入口点
# - 当前向传递结束后，在反向传递中计算图的相关梯度
#%% [markdown]
# #### 动态图优势：如何运行即如何微分
# 
# - 每次迭代，图都会从头开始重建. 这意味着在每次迭代中，都可以使用改变图全部形状和尺寸的任意python控制流表达式.
# - 在开始训练前，不必编码所有可能的路径，就是说:如何运行就如何微分.
#%% [markdown]
# ### 2.3 autograd的就地操作
# 
# 由于autograd的缓存释放与重用机制已经非常高效，所以除非面临非常大的内存压力，否则不鼓励在autograd中使用就地操作.
# 
# 限制就地操作适用性的原因如下：
# 
# - 就地操作可以潜在覆写(overwrite)需要求梯度的变量值
# - 每个就地操作实际上都会要求实现对计算图的重写(rewrite).
#%% [markdown]
# ### 2.4 就地正确性检测
#%% [markdown]
# - 每个tensor都有一个版本计数器，任何导致该tensor变脏的操作，都会使得计数器增加.当Function对象为反向传递保存任何tensor时，这些tensor的计数器也会保存.
# 
# 
# - 每次访问self.saved_tensor都将触发检测，如果发现比保存的值大，将抛出异常.这将会确保在使用就地函数时，不发生任何错误并保证梯度计算的正确性.
#%% [markdown]
# ## 3.Tensor类的autograd函数
#%% [markdown]
# ### 3.1 backward
#%% [markdown]
# - 通过链式法则对图进行微分，计算当前张量关于叶子节点的梯度
# - 如果当前张量是非标量（即含有多于一个元素的张量），计算梯度时需要额外指定gradient参数（看下面的示例）
# - 该函数对叶子节点求梯度会进行累积，如不需要可在调用前对叶子节点已存在的梯度进行清零
#%% [markdown]
# #### 3.1.1 标量梯度

#%%
x=torch.arange(2,4,dtype=torch.float,requires_grad=True)
x


#%%
y=x**2
z=y.sum()
z.backward()##retain_graph=False
x.grad


#%%
type(x.grad)##梯度仍然是Tensor类型


#%%
#z.backward()

#%% [markdown]
# > RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify >retain_graph=True when calling backward the first time.
# 
# __retain_graph=False，因此当反向传递通过图之后，缓存会被释放. 无法进行第二次调用.__

#%%
x.grad.zero_()##如不清零，x的grad就会累积为：8, 12
y=x**2
z=y.sum()
z.backward(retain_graph=True)
x.grad


#%%
z.backward()##因为设置了retain_graph=True，因此这里可以再调用一次
x.grad##梯度累积


#%%
##z.backward() ##retain_graph的生命期只会延续到下一次调用，然后被置为False，导致再次调用抛出异常


#%%
y=x*2
z=y.sum()
z.backward()
x.grad##梯度一直在累积，尽管对应了不同函数


#%%
##对标量值函数反向传递指定gradient
'''
x.grad.zero_()
y=x**2
z=y.sum()
z.backward(gradient=torch.tensor([1,1],dtype=torch.float))
x.grad

'''

#%% [markdown]
# >RuntimeError: invalid gradient at index 0 - expected shape [] but got [2]
# 
# __结论：对于标量函数，gradient不能指定，只能为None__
#%% [markdown]
# #### 3.1.1 多变量函数Jacobian矩阵(  $ f:R^{m}\rightarrow R^{n} $  )

#%%
## 线性函数
x.grad.zero_()
y=2*x##y1=2*x1,y2=2*x2
y.backward(gradient=torch.ones(2,dtype=torch.float))##dy1/dx1=2,dy2/dx2=2
x.grad##求得的结果相当于y的Jacobian矩阵的各列向量之和


#%%
x.grad.zero_()
y=2*x
y.backward(gradient=torch.tensor([1,0],dtype=torch.float))##对于线性函数，这里不指定retain_grad也可以
x.grad##dy1/dx1=2,dy1/dx2=0


#%%
x.grad.zero_()
y.backward(gradient=torch.tensor([0,1],dtype=torch.float))
x.grad##dy1/dx1=2,dy1/dx2=0

#%% [markdown]
# __上面两个合起来就是Jacobian矩阵了__

#%%
## 非线性函数
z=torch.zeros(3,dtype=torch.float).reshape(1,-1)
z[0,0]=x[0]**2+3*x[1]##使用x的元素为z赋值，z的req_grad仍然为True
z[0,1]=x[1]**2+x[0]
z[0,2]=2*x[0]*x[1]
z


#%%
z.requires_grad


#%%
x.grad.zero_()
z.backward(gradient=torch.ones(3).reshape(1,3),retain_graph=True)##这里需要指定retain_grad，否则无法多次backward
x.grad##这个grad是下面的Jacobian矩阵的各列向量的和


#%%
x.grad.zero_()
z.backward(gradient=torch.Tensor([[1,0,0]]),retain_graph=True)
j0=torch.tensor(x.grad)##注意：需要使用tensor包裹一下grad再赋值给j0,这是copy赋值.否则就是引用，到最后Jacobian矩阵的各行向量都一样了
j0


#%%
x.grad.zero_()
z.backward(gradient=torch.Tensor([[0,1,0]]),retain_graph=True)
j1=torch.tensor(x.grad)
j1


#%%
x.grad.zero_()
z.backward(gradient=torch.Tensor([[0,0,1]]),retain_graph=True)
j2=torch.tensor(x.grad)
j2


#%%
torch.zeros(3)


#%%
j=np.vstack((j0.numpy(),j1.numpy(),j2.numpy()))
j##Jacobian矩阵


#%%
x.grad.zero_()
z.backward(gradient=torch.Tensor([[0,0,1]]))
x.grad


#%%
'''
x.grad.zero_()
z.backward(gradient=torch.Tensor([[0,0,1]]))
x.grad

'''

#%% [markdown]
# __由于上面已经取消了retain_grad，因此这里再次backward会抛出异常__
#%% [markdown]
# ### 3.2 detach和detach_
# 
# - detach: 返回当前张量的引用
# - detach\_: 使得当前张量脱离创建它的图，变成一个叶子节点

#%%
x.grad.zero_()
y=2*x
z=y.sum()


#%%
y.is_leaf


#%%
y.requires_grad


#%%
y.detach_()


#%%
y.is_leaf


#%%
y.requires_grad


#%%
z.detach_()


#%%
##z.backward()

#%% [markdown]
# > RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# 
# __注意：detach之后，变成一个叶子节点，无法进行反向传递__

#%%
z=y.sum()##y已经是叶子了，对它的操作输出到z，z仍然是叶子
z.is_leaf


#%%
##先输出操作到z，再detach
y=2*x
z=y.sum()
z.is_leaf


#%%
y.detach_()
z.backward()


#%%
x.grad


#%%
z.is_leaf


#%%
y.is_leaf


#%%
##先detach_，再输出操作到z，此时z也变成了叶子
y=2*x
y.detach_()
z=y.sum()
z.is_leaf


#%%
x_new=x.detach()##detach相当于返回tensor数据的引用
x_new


#%%
x_new[1]=5##对传出的变量进行修改，也会影响到原变量
x


#%%
y=2*x
z=y.sum()
y.is_leaf


#%%
y_new=y.detach()
y.is_leaf

#%% [markdown]
# ### 3.3 register_hook
# 
# - 对中间变量登记该函数，可获得前面变量关于它的梯度

#%%
x.grad.zero_()
y=2*x
z=y.sum()
h = y.register_hook(lambda g:print('grad of z w.r.t y:',g))##这个梯度是z关于y的
z.backward()
h.remove()


#%%
print('grad of z w.r.t x:',x.grad)

#%% [markdown]
# ### 3.4 retain_grad
# 
# - 对于非叶子节点，反向传递之后，并不会保存后续节点关于它的梯度信息
# - retain_grad可以使得非叶子节点再反向传递之后保留梯度信息

#%%
x.grad.zero_()
y=2*x
z=y.sum()
z.backward()


#%%
print(x.grad,'\n',y.grad)


#%%
x.grad.zero_()
y=2*x·
z=y.sum()
y.retain_grad()
z.backward(retain_graph=True)
print(x.grad,'\n',y.grad)


#%%
x.grad.zero_()##对x累积的梯度进行清理
z.backward(retain_graph=True)
print(x.grad,'\n',y.grad)##y也会累积梯度


#%%
x.grad.zero_()
y.grad.zero_()
z.backward()
print(x.grad,'\n',y.grad)

#%% [markdown]
# ## 4. autograd.Function对象
#%% [markdown]
# Function对象记录所有操作历史，并为微分计算定义公式. 具体如下：
# 
# - 在Tensor上执行的每个操作，都会创建一个新的Function对象, 这个对象会执行计算并记录所发生的操作. 所有历史信息保持在由Function对象组成的有向无环图（DAG）中，在这个图上，边（edge）表示依赖关系，比如：input <- output. 然后，当反向传递被调用时，图通过调用每个Function对象的backward()方法，以拓扑序被处理. 然后，把返回的梯度传给下一个Function对象.
# 
# 
# - 通常，用户与Function打交道的唯一方式是创建一个Function子类并定义新的计算操作. 这是扩展autograd的推荐方式.
# 
# 
# - 每个函数对象只被使用一次（在前向传递过程中）.
#%% [markdown]
# ### 4.1 Function的backward和forward方法
#%% [markdown]
# - static forward(ctx, *args, **kwargs): __保存反向传递所需的参数；执行计算并输出__
# 
#     - 执行计算操作. 该函数需要被所有子类重写（override）.
#    
#     - 需要以一个context对象ctx作为第一个参数，接着可以是任意个其他参数（tensor或者其他类型）
# 
#     - context对象用来存储tensor对象.并且可以在反向传递过程中，取回这些参数.
#     
#     
#     
# - static backward(ctx, *grad_outputs)：__读取正向传递保存的参数；计算梯度__
# 
#     - 为计算微分定义公式. 该函数需要被所有子类重写（override）.
#     
#     - 必须以一个context对象作为首个参数，接着是任意个forward()函数返回的值. 并且该函数的返回的tensor个数，需要与传入forward()函数的参数的个数一致. 每个输入参数都是关于给定输出的梯度，每个返回值都是关于相应输入的梯度.
#     
#     - ctx对象可以取回在前向传递中保存的tensor. 它同时有个 ctx.needs_input_grad 属性，该属性是Bool类型的tuple，表示哪个输入需要梯度.
#     例如，如果第一个输入forward函数的参数需要关于输出计算梯度，那么在backward函数就会有：ctx.needs_input_grad[0] = True .

#%%
import torch.autograd as AT


#%%
##自定义乘法

class Mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        r=ctx.save_for_backward(x,y)
        print('in forward\n')
        return x*y
    
    @staticmethod
    def backward(ctx,grad_output):##这个grad_output就是外部调用backward时，传入的gradient
        print('in backward. grad_output is:',grad_output)
        x,y=ctx.saved_tensors
        t=torch.empty(2).fill_(y[0])
        return t,None


#%%
x=torch.arange(2,4,dtype=torch.float,requires_grad=True)
##调用算子：正向传递输出计算结果
z=Mul.apply(x,torch.tensor([2],dtype=torch.float))
z


#%%
##反向传播：计算梯度
z.backward(torch.tensor([1.1,2.2]),retain_graph=True)##这个参数就是grad_output
x.grad


#%%
## 自定义平方
class sq(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x**2).sum()
    
    @staticmethod
    def backward(ctx,grad_output):
        x,=ctx.saved_tensors##注意saved_tensors传回的是个tuple.进行分解之后，再使用.
        print('in backward, grad_output is:',grad_output)
        return 2*grad_output*x


#%%
x


#%%
x.grad.zero_()
y=sq.apply(x)
y.backward()##不输入gradient参数，则默认为1


#%%
x.grad


#%%



