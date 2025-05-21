# 使用PyTorch编程框架实现基于VGG19实现图像分类
## PyTorch背景
PyTorch是由Meta开发并于2017年11月开源的深度学习框架，适用于Python、C++等编程语言，可以用于部署并实施大规模机器学习模型。支持大规模的神经网络模型，并能够支持深度学习算法在CPU、GPU和DLP等硬件平台上的部署。
PyTorch 提供了一系列高性能的 API,方便程序员高效地实现深度学习算法。以目前较为常用的卷积神经网络 VGG 为例，对于每个卷积层，首先输入与权重做卷积运算，然后加上偏置，最后通过非线性激活函数ReLU输出。在第3.1节的实验中使用高级编程语言实现了上述操作，而在PyTorch中则提供了一系列封装好的API，可以方便地实现上述操作。实现 VGG 网络所需的主要函数的使用方法及参数含义如下表所示
| 函数名                         | 功能描述                                       | 参数介绍 |
|------------------------------|----------------------------------------------|----------|
| `torch.nn.Sequential(*args)` | 在 PyTorch 中创建一个时许神经网络容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个 `OrderedDict`。自动实现前向传播 | `*args` 不定参数 |
| `add_module(name, module)`   | 将一个 child module 添加到当前 module。被添加的 module 可以通过 name 属性来获取 | `name`: 子模块的名称，使用给定名称，子模块可以使用给定名称从该模块访问。<br>`module`: 要添加到模块中的子模块。 |
| `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)` | 用来实现二维卷积操作。 | `in_channels`: 输入信号的通道数；<br>`out_channels`: 卷积产生的通道；<br>`kernel_size`: 卷积核的尺寸，可以是单个数也可以是 tuple；<br>`stride`: 卷积步长，默认 1；<br>`padding`: 填充层数，默认 0；<br>`dilation`: 扩张卷积的间距，默认 1；<br>`groups`: 控制分组卷积，默认 1 组；<br>`bias`: 若为 True 添加偏置项，默认 True。 |
| `torch.nn.ReLU(inplace=False)` | 对输入运用 ReLU 函数。 | `inplace`: 选择是否进行覆盖运算。 |
| `torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` | 对于输入信号的输入通道，提供 2 维最大池化操作。 | `kernel_size`: max pooling 窗口大小；<br>`stride`: 移动步长，默认等于 `kernel_size`；<br>`padding`: 输入每一条边填充层数；<br>`dilation`: 一个控制窗口中元素间的参数；<br>`return_indices`: 若为 True，返回池化最大值的索引；<br>`ceil_mode`: 若为 True，使用 ceil 而不是 floor。 |
| `torch.nn.Flatten(start_dim=1, end_dim=-1)` | 将连续的维度范围展平为张量。 | `start_dim`: 第一个维度（默认 1）；<br>`end_dim`: 最后一个维度（默认 -1）。 |
| `torch.nn.Linear(in_features, out_features, bias=True)` | 对输入数据做线性变换：Y = Ax + b | `in_features`: 每个输入样本的大小；<br>`out_features`: 每个输出样本的大小；<br>`bias`: 若为 False，则不添加偏置项，默认 True。 |
| `torch.nn.Softmax(source)` | 对 n 维输入张量运用 Softmax 函数，将输出每个不重叠的 N 维（1D）区间归一为 1。 | 输入: (N, L)；<br>输出: (N, L) |
