
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
import os
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# 原文：https://kevinzakka.github.io/2017/01/18/stn-part2/
# 
# 目录：
# - 动机
# 
# 
# - 池化操作
# 
# 
# - STN结构
#     - Localisation Network
#     
#     - Parametrised Sampling Grid
#     
#     - Differentiable Image Sampling
#     
#     
# - STN应用
#     - 扭曲的MNIST
#     
#     - GTSRB 数据集
#      
#      
# - 总结
# 
# 
# - 参考
#%% [markdown]
# # 动机
# 
# 在执行分类任务时，如果系统对输入变换具有鲁棒性，将会非常有用.
# 
# 就是说，如果对输入进行特定的变换，分类模型仍然能够对这个变换过的输入进行正确分类，就像正确分类未变换的输入一样.
# 
# 分类模型可能遇到的挑战是：
# - 尺度变化
# 
# - 视角变化
# 
# - 变形
# 
# ![image](https://kevinzakka.github.io/assets/stn2/var1.png)
# ![image](https://kevinzakka.github.io/assets/stn2/var2.png)
#%% [markdown]
# 因此，理想的图像分类模型在理论上应该能够从材质和形状中，辨别出物体.
# 
# 再看下面的图像:
# ![image](https://kevinzakka.github.io/assets/stn2/cat2.jpg) ![image](https://kevinzakka.github.io/assets/stn2/cat2_.jpg)
# ![image](https://kevinzakka.github.io/assets/stn2/cat1.jpg) ![image](https://kevinzakka.github.io/assets/stn2/cat1_.jpg)
#%% [markdown]
# 很明显，经过裁剪的图像，对于分类任务来说，更为便捷.
# 
# 因此，如果模型能够通过裁剪和缩放的组合变换，将简化接下来的分类任务.
#%% [markdown]
# # 池化层
# 
# 已经证明，在神经网络架构中使用的池化层，赋予了模型一定程度上的空间不变性.
# 
# 回想一下，池化操作充当下采样功能的机制.
# 
# 随着层次的加深，池化层逐步降低了特征图的空间尺寸，削减了参数量和计算量.
# 
# ![image](https://kevinzakka.github.io/assets/stn2/pool.jpeg)
#%% [markdown]
# 使用尺寸和步长都为2的滤波器，输入由224x224x64变为112x112x64.
# 
# 
# 
# ![image](https://kevinzakka.github.io/assets/stn2/maxpool.jpeg)
#%% [markdown]
# 2x2的最大池化
#%% [markdown]
# __池化层到底怎样提供了空间不变性?__
# 
# 池化背后的思想是，把一个复杂的输入，切分为数小块，从这些复杂的小块中，采集信息以产生较简单的部分，并使用这些较简单的部分来描述输出.
# 
# 举例来说，我们有3张数字7的图像，每张都是不同的方向. 
# 
# 每张图像网格上的池化层，将会检测到7，而不必考虑它在网格中的位置. 这样通过聚合像素值，我们将会捕获到近似相同的信息.
#%% [markdown]
# 但是池化有几个缺点，使得它并不是那么理想的操作. 其中一点就是，池化具备毁灭性. 
# 
# 使用时，池化丢弃了75%的特征激活，这就意味着一定会丢弃确切的位置信息.
# 
# 现在你也许会有疑问，为何先前说池化赋予了网络空间不变的鲁棒性，而在这里变成了缺点.
# 
# 因为位置信息在视觉识别任务中，是无价的.
# 
# 思考上面猫的分类任务.
# 
# 知道胡须相对于嘴鼻的位置信息是重要的，但是如果使用最大池化，那么这些信息就会被丢弃.
# 
# 另一个关于池化的限制是，它是局部的并且是预定义的. 由于感受野的尺寸较小，池化的效果只有在更深的层中才有效，使得来自较大输入的中间的特征图遭受到扭曲.
# 
# 并且需要记住，我们不能任意增大感受野的尺寸，那样将会过于激进的对特征图进行下采样.
# 
# 所以主要的问题就是卷积网络对于相对较大的输入扭曲，不具备空间不变性. 这个问题主要是因为受限的、预定义的池化机制造成的.
# 
# 因此STN就闪亮登场了！
# 
# >卷积神经网络中的池化操作就是个巨大失误，并且它能工作的如此之好，更是个大灾难
# >
# >-Hinton
#%% [markdown]
# # STN
# 
# STN通过为卷积神经网络提供显式的空间变换能力，来解决上述的问题.
# 
# 它具有3个很好的性质：
# - 模块性：仅需相对较小的调整，STN就可插入到现有架构的任何地方.
# - 可微性：注入STN的模块，可以进行端到端的训练，并使用反向传播训练STN.
# - 动态性：STN在每个输入样本的特征图上执行活动的空间变换.与之相比，池化层对所有输入执行相同的操作.
# 
# 因此STN在各个方面都比池化要高级.
# 
# ![image](https://kevinzakka.github.io/assets/stn2/stn_arch.png)
# 
# 如上图所示，STN由3部分组成:localisation network，grid genertator, sampler. 
# 
# 在深入了解细节之前，回想一下仿射变换的3个步骤：
# 
# - 首先，创建一个由(x,y)组成的坐标网格.例如，对于一个400x400的灰阶图，我们创建一个具有相同维度的坐标网格，使得x均匀位于[0,W-1], y均匀位于[0,H-1]
# - 然后，对生成的网格施加变换矩阵
# - 最后，用变换后的网格对原图像进行采样，并使用合适的插值方法
# 
# 记住，我们不能立即对原图进行仿射变换. 首先要创建网格（内容是坐标），然后对它进行变换（内容是变换后的坐标），然后使用变换后的网格对图像采样（用变换后网格内容作为索引）.
#%% [markdown]
# ## Localisation Network
# 
# Localisation Network的目标是分离出仿射变换的参数$\Theta$，该参数将作用于输入的特征图.
# 
# 我们的Localisation Network的定义如下：
# 
# - input: 形状为(H,W,C)的特征图U
# - output:形状为(6,)的变换矩阵$\Theta$
# - 架构：全连接网络或者卷积网络
# 
# 当我们训练我们的网络时，希望网络能输出越来越精确的参数$\Theta$.
# 
# 那么精确是什么意思？
# 
# 想象一下，90度逆时针旋转的数字7的图像. 比如在经过2轮训练之后，localisation network可以输出一个变换矩阵，该矩阵能对图像进行45度顺时针旋转. 再经过5轮之后，LN终于学得一个能进行90度顺时针旋转的变换矩阵. 这样我们输出的图像看着就像一个标准的数字7，接下来的分类就会更容易.
# 
# 从另一方面看，LN学得了对每个输入样本进行变换的参数知识，并把这些知识存储于它的层权重之中.
#%% [markdown]
# ## Parametrised Sampling Grid
# 
# 网格生成器的任务是输出参数化的采样网格，该采样网格由坐标点的集合组成，通过它对输入图进行采样，以获得期望的变换输出.
# 
# 具体来说，网格生成器首先创建归一化的网格，该网格与输入图像U具有相同的尺寸：(H,W)，就是覆盖了整个输入特征图的下标$(x^{t},y^{t})$，这里的标注t表示输出特征图的目标坐标.
# 
# 然后，由于我们对网格进行仿射变换并需要进行平移，因此需要为坐标向量添加一行，以构成齐次坐标.
# 
# 最后，把参数$\Theta$重整为2x3的变换矩阵，并进行如下变换：
# $\begin{bmatrix}
# x^{s}\\
# y^{s}
# \end{bmatrix}=\begin{bmatrix}
# \theta_{11}&\theta_{12}&\theta_{13}\\
# \theta_{21}&\theta_{22}&\theta_{23}
# \end{bmatrix}\begin{bmatrix}
# x^{t}\\
# y^{t}\\
# 1\end{bmatrix}$
# 
# 
# 列向量$\begin{bmatrix}
# x^{s}\\
# y^{s}
# \end{bmatrix}$将告诉我们如何从输入中采样，以获得期望的输出.
# 
# 但是如果这些坐标是分数怎么办？这就需要双线性插值出场了.
#%% [markdown]
# ## 可微分的图像采样
# 
# 由于双线性插值是可微分的，因此非常适合手边的任务. 具备了输入特征图和参数化的采样网格，通过双线性采样来获得输出的特征图，形状为：(H1,W1,C1). 注意这意味着通过指定采样网格的形状，我们可以执行下采样和上采样.
# 
# 并不是一定要使用双线性插值，可以使用其他的采样核，但是重要的是，这些采样核一定是可微的，以使得损失函数的梯度能够反向回流到STN网络.
# 
# ![image](https://kevinzakka.github.io/assets/stn2/transformation.png)
# 
#%% [markdown]
# 左图：恒等变换. 右图：仿射变换.
# 
# 所以主要归结为两个重要的概念：一个仿射变换跟着一个双线性插值. 我们使用网络学习最优的仿射变换参数，将会使得分类任务极大成功，并且这一切都是自动执行的.
#%% [markdown]
# # 应用
# 
# ## 扭曲的MNIST
# 
# 这里是一个使用STN作为FC首层的网络，用来对扭曲的MNIST进行分类：
# 
# 注意它是怎样学习变为我们期望的理论上的鲁棒图片分类模型：
# 
# 通过放大并消除背景杂斑，对输入进行标准化，以便于进行分类.
# 
# ![image](https://kevinzakka.github.io/assets/stn2/mnist.png)
# 
# 动画在此：https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view
#%% [markdown]
# # 总结
# 
# 略.
# 
# # 参考
# 
# 略.

#%%



