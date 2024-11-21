import imp
from lib2to3.pytree import LeafPattern
import torch
import random
from d2l import torch as d2l

# 模拟样本数据集
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

# print('features:',features[0],'\nlabel:',labels[0])

# 读取小批量样本数据
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    # 随机读取样本
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]  

batch_size = 10

# 小批量数据预览
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义线性回归模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b

# 定义损失函数
def squared_loss(y_hat,y):  
    # y_hat 预测值
    return (y_hat-y.reshape(y_hat.shape))**2/2
   
# 优化算法（随机梯度降）
# lr:学习速率
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练过程
lr = 0.3
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y) # 批量损失
        l.sum().backward() # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        # 更新后的参数在整个数据集上跑一遍
        train_l=loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')