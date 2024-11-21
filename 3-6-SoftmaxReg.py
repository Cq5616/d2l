
import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# softmax 函数
# X: a batch of output (batch_size*type_num)，每行是一组输出
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True) # dim=1 按行求和 partition(batch_size*1)
    return X_exp/partition # 广播机制 return:(batch_size*type_num)

# 检验softmax
# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X,X_prob, X_prob.sum(1))

# softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # 使⽤reshape函数将每张原始图像展平为向量

# 损失函数
def cross_entropy(y_hat, y):
    # 示例
    # y = torch.tensor([0, 2]) # 每个样本的类别标签
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # y_hat 每个样本，每种类别的概率 
    # y_hat[[0, 1], y]
    return - torch.log(y_hat[range(len(y_hat)), y]) # 通过⼀个运算符[]完成计算，避免使⽤for循环

# 优化方法 随机梯度降
def updater(batch_size):
    return d2l.sdg([W,b],lr,batch_size)

# 检验损失函数
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(cross_entropy(y_hat, y))

# 分类精度（衡量训练效果，但不能用做损失函数）
def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 通用方法，分类问题的精度评估
def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator: #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # ?
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 定义一个在动画中绘制数据的实用程序类
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)



# 模型训练（通用方法）
def train_epoch(net,train_iter,loss,updater):
    """训练一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        nn.train()
    # 累计指标：训练损失总和，训练准确度总和，样本数
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        # 
        if isinstance(updater,torch.optim.Optimizer):
            # 使用PyTorch内置优化器
            updater.zero_grad()
            l.mean().backwarc()
            updater.step()
        else:
            # 使用定制的优化器
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    # 返回训练损失和训练精度
    return metric[0]/metric[2],metric[1]/metric[2]

def train(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics=train_epoch(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# run
lr=0.1



num_epochs=10
train(net,train_iter,test_iter,cross_entropy,num_epochs,updater)