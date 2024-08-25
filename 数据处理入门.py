# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:23:58 2024

@author: 86150
"""

"""
    数据操作
"""


import torch

"使用 arange 创建一个行向量 x"
x = torch.arange(12)
print(x)

"通过张量的shape属性来访问张量"
x.shape

"形状的所有元素乘积，可以检查它的大小（size）"
x.numel()

"要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数"
X = x.reshape(3, 4)
X

"创建一个形状为（2,3,4）的张量，其中所有元素都设置为0"
torch.zeros((2, 3, 4))

"创建一个形状为(2,3,4)的张量，其中所有元素都设置为1"
torch.ones((2, 3, 4))

"创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。"
torch.randn(3, 4)

"提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。 在这里，最外层的列表对应于轴0，内层的列表对应于轴1"
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

"常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算。 我们可以在同一形状的任意两个张量上调用按元素操作"
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

"“按元素”方式可以应用更多的计算"
torch.exp(x)

"多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量"
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

"通过逻辑运算符构建二元张量"
X == Y

"对张量中的所有元素进行求和，会产生一个单元素张量"
X.sum()

"在某些情况下，即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作"
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b

a + b

"索引和切片"
X[-1], X[1:3]
X[1, 2] = 9
X

"节省内存"
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))


"深度学习框架定义的张量转换为NumPy张量（ndarray）"
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)


"""
    数据预处理
"""

import os

"写入数据"
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    
import pandas as pd

data = pd.read_csv(data_file)
print(data)

"处理缺失值"
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

"转换为张量格式"
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y

"""
    线性代数
"""
import torch

"标量"
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y

"向量"
x = torch.arange(4)
x

"矩阵"
A = torch.arange(20).reshape(5, 4)
A

A.T

"张量"
X = torch.arange(24).reshape(2, 3, 4)
X

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B

"Hadamard积"
A * B

a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

"降维"
x = torch.arange(4, dtype=torch.float32)
x, x.sum()

A.shape, A.sum()

# 沿着轴加总
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape

A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape

A.sum(axis=[0, 1])  # 结果和A.sum()相同

A.mean(), A.sum() / A.numel()

A.mean(axis=0), A.sum(axis=0) / A.shape[0]

"非降维求和"
sum_A = A.sum(axis=1, keepdims=True)
sum_A

A / sum_A

A.cumsum(axis=0)

"点积"
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

"矩阵-向量积"
A.shape, x.shape, torch.mv(A, x)

"矩阵-矩阵乘法"
B = torch.ones(4, 3)
torch.mm(A, B)

"范数"
u = torch.tensor([3.0, -4.0])
torch.norm(u)

torch.abs(u).sum()

torch.norm(torch.ones((4, 9)))


"""
    微积分
"""

"导师与微分"
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
    
    
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
    
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


"""
    概率论
"""
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()

multinomial.Multinomial(10, fair_probs).sample()

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();

import torch

print(dir(torch.distributions))

help(torch.ones)

torch.ones(4)