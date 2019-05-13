
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
import torch.distributions as distributions
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 1.概率分布包：torch.distributions
# 
# 概率分布包，torch.distributions，包括可参数化的概率分布和采样函数.
# 
# 这将允许为优化问题构造随机计算图(stochastic computation graphs)和随机梯度估计(stochastic gradient estimators).
# 
# 这个包大体上遵循 TensorFlow Distributions 的设计.
# 
# 不可能经过随机样本直接进行反向传播. 但是有两个主要办法通过创建代理函数，可以使得反向传播通过.
# 
# 这就是：得分函数估计（score funciton estimator）、似然比估计（likehood ratio estimator）、增强和路径导数估计（REINFORCE and pathwise derivative estimator）.
# 
# 作为增强学习中策略梯度方法的基础，增强（REINFORCE）在增强学习会经常被看到. 同时，路径导数估计作为变分自动编码器中重参数化的技巧，也在其中经常使用.
# 
# 同时，得分函数仅需要样本函数的值；路径导数需要函数的导数.
# 
# 下面在增强学习示例中讨论 得分函数 和 路径导数.
#%% [markdown]
# ## 1.1 得分函数 Score Function
# 
# 当概率密度函数关于它的参数可微时，仅需要sample()和log_prog()就可实现REINFORCE：
# 
# $\Delta \theta=\alpha r \frac{\partial \log p( a\mid \pi^{\theta}(s))}{\partial \theta}$
# 
# 其中,$\theta$表示全部参数，$\alpha$表示学习率, $r$表示回报.
# 
# $p( a\mid \pi^{\theta}(s))$表示在状态s时给定策略$\pi^{\theta}$，采取行动a的概率.
#%% [markdown]
# 在实际中，我们会从网络的输出中采样行动，把这个行动作用于一个环境，然后使用log_prob来构造等价的损失函数.
# 
# 需要注意，我们使用一个负号，因为优化器使用梯度下降，而上述的假设是梯度上升.
# 
# 通过使用分类策略，实现REINFORCE的代码如下：
# 
# 见:https://pytorch.org/docs/stable/distributions.html#score-function
#%% [markdown]
# ## 1.2 路径导数 Pathwise Derivative
# 
# 另一种实现这种随机梯度或策略梯度的方法，是通过resample()方法，使用重参数化技巧. 因为在resample()方法中，参数化的随机变量，可以通过无参数随机变量的参数化确定性函数来构造.(大概是通过简单抽样来获得复杂的抽样).重参数的样本因此变得可微分.
# 
# 代码示例见:https://pytorch.org/docs/stable/distributions.html#pathwise-derivative
#%% [markdown]
# # 2.分布函数类
# 
# ## 2.1 所有概率分布的抽象基类：class torch.distributions.distribution.Distribution
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution
#%% [markdown]
# ## 2.2 指数族分布的抽象基类：class torch.distributions.exp_family.ExponentialFamily
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.exp_family.ExponentialFamily
#%% [markdown]
# ## 2.3 bernoulli分布
# 
# 以指数族分布为基类.
# 
# 创建一个以probs或logits为参数的bernoulli分布: 样本为0或者1，以probs概率采样到1 ，以1-probs概率采样到0.
# 
# 
# 即概率分布为：
# 
# $p(x \mid u) = u^{x}(1-u)^{1-x}, x=0或1, 0 \le u \le 1 且p(x = 1) = u.$
# 
# __它的广义形式为 范畴分布 Categorical Distribution，即广义的bernoulli分布.__

#%%
##构造p=0.5的bernoulli分布
bern=distributions.bernoulli.Bernoulli(0.5)


#%%
##单个采样
bern.sample()


#%%
##批采样
bern.sample((5,))


#%%
##均值
bern.mean


#%%
##标准差
bern.stddev


#%%
##方差
bern.variance


#%%
##熵：平均信息量
bern.entropy()


#%%
bern.param_shape

#%% [markdown]
# ## 2.4 beta分布
# 
# 以指数族为基类.
# 
# 以concentration1($\alpha$，第一个参数) 和 concentration0($\beta$，第二个参数)为参数构建.

#%%
beta=distributions.beta.Beta(0.5,0.5)


#%%
beta.sample()


#%%
beta.sample((2,3))


#%%
beta.mean


#%%
beta.stddev


#%%
beta.variance


#%%
beta.entropy()


#%%
beta.concentration0


#%%
beta.concentration1

#%% [markdown]
# ## 2.5 二项分布
# 
# 
# 以Distribution为基类.
# 
# - 第一个参数为bernoulli试验次数
# - 第二个参数为probs
# - 第三个参数为logits
# 
# 
# 它的概率分布为：
# 
# $p(x \mid N, u) = C_{N}^{k}p^{k}(1-p)^{N-k}$
# 
# __它的广义形式为 多项分布 Multinomial Distribution.__

#%%
binom=distributions.binomial.Binomial(10,0.3)
binom.sample((5,))


#%%
binom.mean


#%%
binom.variance


#%%
binom2=distributions.binomial.Binomial(10,torch.tensor([0.2,0.6]))##两个相互独立的试验，分布进行10次
binom2.sample()


#%%
## Multinomial Distribution 多项分布，具体见2.19.
multn=distributions.multinomial.Multinomial(10,torch.tensor([2.4,6.5,0.3]))##内部归一化
multn.sample((5,))

#%% [markdown]
# ## 2.6 范畴分布 Categorical Distribution，又称为广义 bernoulli 分布
# 
# 
# 
# __它是bernoulli分布的推广.__
# 
# $x={x_{1},x_{2},...,x_{k}}$, x的分量采用one-of-k的方式（即2.22的OneHot形式）表示：
# 
# 且$x_{k}=0或1,\sum_{k}^{K}x_{k}=1$，即x的分量中只有一个为1，其他都为0. 且$p(x_{k}) = u_{k},u_{k}\ge 0, \sum_{k}^{K}u_{k}=1$
# 
# 则  __范畴分布 categorical distribution的概率分布为：__
# 
# 
# $p(x \mid u)=\prod_{k}^{K}u_{k}^{x_{k}}$
#%% [markdown]
# __注意：输入的概率张量，不要求和为1，只要求非负. 它内部会归一化处理__

#%%
cat=distributions.categorical.Categorical(torch.tensor([0.3,1,0.6]))##内部进行归一化
cat.sample((10,))


#%%
cat.mean


#%%
cat.stddev

#%% [markdown]
# ## 2.7 柯西分布 Cauchy Distribution

#%%
cauchy=distributions.cauchy.Cauchy(0,1)
cauchy.sample((3,))


#%%
##累积分布函数
cauchy.cdf(0)


#%%
cauchy.mean

#%% [markdown]
# ## 2.8 Chi2分布
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.chi2.Chi2
#%% [markdown]
# ## 2.9 迪利克雷分布 Dirichlet Distribution
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.dirichlet.Dirichlet
#%% [markdown]
# ## 2.10 指数分布 Exponential Distribution
# 

#%%
expo=distributions.exponential.Exponential(0.2)
expo.mean


#%%
expo.sample((3,))


#%%
expo.rsample((2,))


#%%
expo.cdf(5)

#%% [markdown]
# ## 2.11 Fisher-Snedecor 分布
#%% [markdown]
# ## 2.12 Gamma分布
# 
# - 第一个参数为concentration，即$\alpha$
# 
# - 第二个参数为比率，即$\beta$

#%%
gamma=distributions.gamma.Gamma(1,1)
gamma.sample((3,))

#%% [markdown]
# ## 2.13 几何分布 直到第k次成功的概率
# 
# 返回的样本值为[0,inf)

#%%
geo=distributions.geometric.Geometric(0.7)
geo.sample()

#%% [markdown]
# ## 2.14 Gumbel 分布
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.gumbel.Gumbel
#%% [markdown]
# ## 2.15 HalfCauchy 分布
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_cauchy.HalfCauchy
#%% [markdown]
# ## 2.16 半正太分布 HalfNormal Distribution
# 
# X ~ Normal(0, scale)
# 
# Y = |X| ~ HalfNormal(scale)

#%%
hNorm=distributions.half_normal.HalfNormal(0.5)
hNorm.sample((3,))

#%% [markdown]
# ## 2.17 拉普拉斯分布 Laplace Distribution

#%%
lap=distributions.laplace.Laplace(0,0.5)
lap.sample((3,))


#%%
lap.mean


#%%
lap.variance


#%%
lap.cdf(0)

#%% [markdown]
# ## 2.18 对数正太分布 LogNormal Distribution

#%%
logNormal=distributions.log_normal.LogNormal(0,2)
logNormal.sample((3,))


#%%
logNormal.mean

#%% [markdown]
# ## 2.19 多项分布 Multinomial Distribution
# 
# 
# __它是二项分布 binominal distribution的推广.__
# 
# 
# 
# $x={x_{1},x_{2},...,x_{k}}$, x的分量采用one-of-k的方式（即2.22的OneHot形式）表示：
# 
# 且$x_{k}=0或1,\sum_{k}^{K}x_{k}=1$，即x的分量中只有一个为1，其他都为0. 且$p(x_{k}) = u_{k},u_{k}\ge 0, \sum_{k}^{K}u_{k}=1$
# 
# 则：
# $p(x \mid u)=\prod_{k}^{K}u_{k}^{x_{k}}$，这就是  __范畴分布 categorical distribution的概率分布.__
# 
# 那么N个样本的数据集：$x_{1},x_{2},..x_{n}$的似然函数为：
# 
# $p(D \mid u) = \prod_{k}^{K}u_{k}^{m_{k}}，其中m_{k}=\sum_{n}^{N}x_{nk}$
# 
# 
# 那么Multinomial Distribution定义为：
# 
# $Mult(m_{1},m_{2},...,m_{k} \mid N,u) = \frac{N!}{m_{1}!m_{2}!...m_{k}!}\prod_{k}^{K}u_{k}^{m_{k}}$，这是  __多项分布 multinomial distribution的概率分布.__
# 
# 其中，$\sum_{k}^{K}m_{k}=N$
# 
# 这个和上面的范畴分布区别，这里需要给出试验次数为参数. 
# 对比 2.6
# 
# 
# __注意：输入的概率张量，不要求和为1，只要求非负. 它内部会归一化处理__

#%%
multn=distributions.multinomial.Multinomial(10,torch.tensor([2.4,6.5,0.3]))
multn.sample((5,))

#%% [markdown]
# ## 2.20 多变量的正太分布 MultivariateNormal Distribution
# 
# 输入参数为均值向量与协方差矩阵.

#%%
mean=torch.zeros(2)
std=torch.eye(2)
mn=distributions.multivariate_normal.MultivariateNormal(mean,std)


#%%
mn.sample((3,))

#%% [markdown]
# ## 2.21 正太分布 Normal Distribution

#%%
norm=distributions.normal.Normal(0,1)
norm.sample((3,))

#%% [markdown]
# ## 2.22 OneHot Categorical 分布
# 
# 生成 one-of-k 样本的分布
# 
# __注意：输入的概率张量，不要求和为1，只要求非负. 它内部会归一化处理__

#%%
oh=distributions.one_hot_categorical.OneHotCategorical(torch.tensor([0.3,0.4,0.2,0.1]))
oh.sample((3,))

#%% [markdown]
# ## 2.23 帕累托分布 Pareto Distribution
# 
# 见： https://pytorch.org/docs/stable/distributions.html#torch.distributions.pareto.Pareto
#%% [markdown]
# ## 2.24 泊松分布 Possion Distribution

#%%
poi=distributions.poisson.Poisson(2)
poi.mean


#%%
poi.variance


#%%
poi.sample((3,))

#%% [markdown]
# ## 2.25 RelaxedBernoulli 分布
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli
#%% [markdown]
# ## 2.26 RelaxedOneHotCategorical
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical
#%% [markdown]
# ## 2.27 t分布
# 
# - 第一个参数df为自由度

#%%
td=distributions.studentT.StudentT(3)
td.sample((3,))

#%% [markdown]
# ## 2.28 分布变换
# 
# 即对基本分布施加特定的变换，生成新的分布.
# 
# 以下分布都是经过变换生成的新的分布：
# 
# - Gumbel
# 
# - HalfCauchy
# 
# - HalfNormal
# 
# - LogNormal
# 
# - Pareto
# 
# - RelaxedBernoulli
# 
# - RelaxedOneHotCategorial
#%% [markdown]
# ### 示例：对均匀分布施加sigmoid变换+仿射变换，形成Logistic分布

#%%
base_distribution=distributions.uniform.Uniform(0,1)
transfroms=[distributions.transforms.SigmoidTransform().inv,distributions.transforms.AffineTransform(2,3)]
logistic=distributions.transformed_distribution.TransformedDistribution(base_distribution,transfroms)


#%%
logistic.sample((3,))

#%% [markdown]
# ## 2.29 均匀分布
# 
# 略.
#%% [markdown]
# # 3. KL 散度
# 
# 
# ## 3.1 计算两个分布的KL散度
# 计算两个分布的KL散度，即：
# 
# $KL(p\mid \mid q) = \int p(x) \log \frac{p(x)}{q(x)}$
# 
# 输入的p和q，需要为Distribution对象.

#%%
##计算指数分布与正太分布的KL散度
expon=distributions.exponential.Exponential(0.5)
norm=distributions.normal.Normal(0,1)
distributions.kl.kl_divergence(expon,norm)


#%%
distributions.kl.kl_divergence(norm,expon)


#%%
distributions.kl.kl_divergence(norm,norm)

#%% [markdown]
# ## 3.2 注册KL散度计算的装饰器
# 
# 见：https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.register_kl
#%% [markdown]
# # 4. 变换
# 
# - transforms.Transform ：变换的抽象基类
# 
# 
# - transforms.ComposeTransform：组合变换类
# 
# 
# - transforms.ExpTransform：指数变换类
# 
# 
# - transforms.PowerTransform：幂变换类
# 
# 
# - transforms.SigmoidTransform：Sigmoid变换类
# 
# 
# - transforms.AbsTransform： 绝对值变换类
# 
# 
# - transforms.AffineTransform：仿射变换类
# 
# 
# - transforms.SoftmaxTransform：softmax变换类
# 
# 
# - transforms.StickBreakingTransform：
# 
# 
# - transforms.LowerCholeskyTransform
#%% [markdown]
# # 5. 约束 Constraints
# 
# constraints.Constraint: 约束对象的抽象基类. 约束对象表示了一片区域，在这片区域上变量是有效的，因而可以进行优化.
# 
# 见：https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints
#%% [markdown]
# # 6. 约束注册表 Constraint Registry
# 
# Pytorch提供了两个链接Constraint对象和Transform对象的Constraint Registry 对象.
# 
# 这些对象均要求constraints输入，并返回 trnaforms，但是在双射性方面有不同的保证.
# 
# 见：https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraint_registry

#%%



