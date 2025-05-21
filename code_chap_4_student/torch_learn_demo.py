import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# # torchvision
# 作用：torchvision 是 PyTorch 的官方视觉库，提供以下主要功能：

# 数据集（Datasets）：预加载常用视觉数据集（如 MNIST、CIFAR-10、ImageNet 等），通过 torchvision.datasets 模块实现。

# 模型（Models）：预训练的经典深度学习模型（如 ResNet、VGG、AlexNet 等），通过 torchvision.models 模块调用。

# 图像处理工具（Transforms）：通过 torchvision.transforms 提供数据增强和预处理方法。

# 工具函数：如图像网格生成（make_grid）、视频处理等。

# 典型用途：
# 快速加载数据集、调用预训练模型或实现图像处理流程。
import torchvision.transforms as transforms
# torchvision.transforms
# 作用：专门用于图像预处理和数据增强的模块，提供了一系列可组合的变换操作：

# 归一化：如 Normalize(mean, std)

# 尺寸调整：如 Resize(), CenterCrop()

# 颜色变换：如 Grayscale(), ColorJitter()

# 几何变换：如 RandomHorizontalFlip(), RandomRotation()

# 类型转换：如 ToTensor()（将 PIL 图像或 NumPy 数组转为张量）

# 典型用途：
# 在训练前对图像进行标准化、增强数据多样性，或适配模型输入格式。
from torch.utils.data import DataLoader

# 1. nn.Module 和 nn.Sequential 的关系
# 🔹 nn.Module 是 PyTorch 中所有神经网络模块的基类
# 自定义模型、现成的层（如 nn.Linear、nn.Conv2d）、容器（如 nn.Sequential、nn.ModuleList）等都继承自它。

# 它定义了统一的接口，比如 forward()、train()、eval()、parameters() 等。

# 🔹 nn.Sequential 是 nn.Module 的子类
# 它是一个模块容器，内部自动按顺序调用每个子模块的 forward()。

# 它可以将多个层按顺序组合，构成一个小网络。


# 1. 定义网络结构，继承自nn.Module，也就是model本身是一个模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # 将图像拉平成一维
            nn.Linear(28*28, 256),      # 第一个全连接层
            nn.ReLU(),                  # 激活函数
            nn.Dropout(0.2),            # Dropout 层
            nn.Linear(256, 10)          # 输出层（10类）
        )

    def forward(self, x):
        return self.net(x)

# 2. 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 3. 数据加载与预处理
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, 
                                           transform=transform #将transform转换绑定到整个数据集。实现自动tensor化
                                           )
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# 4. 设置设备、模型、损失函数、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练过程
for epoch in range(num_epochs):
    # 设置模型为训练模式
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        #根据模型正向传播，计算结果
        output = model(batch_x)
        #根据计算结果与正确值计算损失
        loss = loss_fn(output, batch_y)
        #梯度清零
        optimizer.zero_grad()
        #反向传播，根据损失计算参数梯度
        loss.backward()
        #参数更新--根据梯度
        optimizer.step()
        #将本epoch的损失统计到总损失中
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 6. 测试过程
#设置模型为测试模式，关闭训练特有的行为（如 Dropout、BatchNorm 的随机性）
model.eval()
correct = 0
total = 0
with torch.no_grad():#禁用梯度计算，加速评估
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        _, predicted = torch.max(output, 1)#返回张量沿指定维度的最大值及其索引（也就是类别）
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
