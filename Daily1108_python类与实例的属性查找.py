#%% [markdown]
# # descriptor以及类和实例的属性查找
# 
# 参考：
# 
# http://www.cnblogs.com/xybaby/p/6266686.html
# 
# 
# https://www.cnblogs.com/xybaby/p/6270551.html

#%%
class DataDescriptor(object):
    def __init__(self, init_value):
        self.value = init_value

    def __get__(self, instance, typ):
        return 'DataDescriptor __get__'

    def __set__(self, instance, value):
        print ('DataDescriptor __set__')
        self.value = value

class NonDataDescriptor(object):
    def __init__(self, init_value):
        self.value = init_value

    def __get__(self, instance, typ):
        return('NonDataDescriptor __get__')

class Base(object):
    dd_base = DataDescriptor(0)
    ndd_base = NonDataDescriptor(0)


class Derive(Base):
    dd_derive = DataDescriptor(0)
    ndd_derive = NonDataDescriptor(0)
    same_name_attr = 'attr in class'

    def __init__(self):
        self.not_des_attr = 'I am not descriptor attr'
        self.same_name_attr = 'attr in object'

    def __getattr__(self, key):
        return '__getattr__ with key %s' % key

    def change_attr(self):
        self.__dict__['dd_base'] = 'dd_base now in object dict '
        self.__dict__['ndd_derive'] = 'ndd_derive now in object dict '


#%%
b = Base()
d = Derive()

#%% [markdown]
# ## 1. 查看类__dict__与实例__dict__
#%% [markdown]
# ### 1.1 类dict
# - 预定义的变量或函数：doc，module等
# - 自定义的函数
# - init外的所有变量

#%%
Derive.__dict__

#%% [markdown]
# ### 1.2 实例dict
# - init内的所有变量

#%%
d.__dict__

#%% [markdown]
# ## 2. 实例属性查找
#%% [markdown]
# 如果一个descriptor只实现了__get__方法，我们称之为non-data descriptor， 如果同时实现了__get__ __set__我们称之为data descriptor .
# 
# 
# obj = Clz(), 那么obj.attr 顺序如下：
# 
#     （1）如果“attr”是出现在Clz或其基类的__dict__中， 且attr是data descriptor， 那么调用其__get__方法, 否则
# 
#     （2）如果“attr”出现在obj的__dict__中， 那么直接返回 obj.__dict__['attr']， 否则
# 
#     （3）如果“attr”出现在Clz或其基类的__dict__中
# 
#         （3.1）如果attr是non-data descriptor，那么调用其__get__方法， 否则
# 
#         （3.2）返回 __dict__['attr']
# 
#     （4）如果Clz有__getattr__方法，调用__getattr__方法，否则
# 
#     （5）抛出AttributeError 

#%%
d.dd_base


#%%
d.ndd_derive


#%%
d.not_des_attr


#%%
##等同于
getattr(d,'not_des_attr')


#%%
d.no_exists_key


#%%
d.same_name_attr##与类属性同名的实例属性


#%%
Derive.same_name_attr


#%%
##调用实例的change_attr
d.change_attr()


#%%
d.dd_base##这里实例调用的仍然是类定义的dd_base. 因为对于descriptor,类的查找顺序优先于实例


#%%
d.__dict__


#%%
d.ndd_derive##调用的实例子集的ndd_derive了. 这时候又变为实例查找顺序优先于类的顺序

#%% [markdown]
# ## 3. 类属性查找
# 
# 前面提到过，类是元类（metaclass）的实例，因此也是对象，所以类属性的查找顺序基本同上。
# 
# 区别在于第二步，由于Clz可能有基类，所以是在Clz及其基类的__dict__”查找“attr，注意这里的查找并不是直接返回clz.__dict__['attr']。具体来说，这第二步分为以下两种情况：
# 
# 　　（2.1）如果clz.__dict__['attr']是一个descriptor（不管是data descriptor还是non-data descriptor），都调用其__get__方法
# 
# 　　（2.2）否则返回clz.__dict__['attr']

#%%



