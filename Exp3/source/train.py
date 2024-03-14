import torch
from d2l import torch as d2l
from IPython import display
from torch import nn
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

# 大批量训练准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('b-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
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
        
    def save_fig(self, filename):
        # 保存当前动画的最终状态到文件
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.savefig(filename)

batch_size = 256
trans = torchvision.transforms.ToTensor()



mnist_train = torchvision.datasets.FashionMNIST(
    root="G:\大学课程\大三下\深度学习\实验专辑\code\Exp2\data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="G:\大学课程\大三下\深度学习\实验专辑\code\Exp2\data", train=False, transform=trans, download=True)

train_loader = Data.DataLoader(mnist_train, batch_size, shuffle=True,
                                    num_workers=1)
test_loader =Data.DataLoader(mnist_test, batch_size, shuffle=False,
                                    num_workers=1)


loss = nn.CrossEntropyLoss(reduction='none')





class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MyMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
def evaluate_accuracy(model, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(model, torch.nn.Module):
        model.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(d2l.accuracy(model(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(model, train_loader, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(model, torch.nn.Module):
        model.train().to(device)
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for X, y in train_loader:
        # 计算梯度并更新参数
        X = X.to(device)
        y = y.to(device)
        # 把X的第一维放到最后一维
        X = X.reshape(-1, 784)
        y_hat = model(X)
        l = loss(y_hat, y).to(device)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

num_epochs, lr = 100, 0.1
num_inputs, num_outputs, num_hiddens = 784, 10, [256,64]

def train_ch3(model, train_loader, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(model, train_loader, loss, updater)
        test_acc = evaluate_accuracy(model, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    animator.save_fig(f'../media/MLP_h2_{num_epochs}.png')
    train_loss, train_acc = train_metrics



model = MyMLP(num_inputs, num_hiddens[0], num_hiddens[1], num_outputs)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

train_ch3(model,train_loader,test_loader,loss,num_epochs,optimizer)

