import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn # nn是神经⽹络的缩写

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
batch_size = 10
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))

# 定义模型（全连接层）
net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 使用替换⽅法normal_和fill_来重写参数值。
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss() # 返回所有样本损失的平均值
# 优化算法（随机梯度降）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    w=net[0].weight.data
    b=net[0].bias.data
    print(f'w:{w},b:{b}')
    print(f'epoch {epoch+1}, loss {l:f}')

