import time
import torch
import torch.nn as nn

# 2、线性回归范例：

# 准备数据：
n = 1000000
x = 10 * torch.rand([n, 2]) - 5.0
w0 = torch.tensor([[2.0, -3.0]])
b0 = torch.tensor([[10.0]])
y = x @ w0.t() + b0 + torch.normal(0.0, 2.0, size=[n, 1])  # 增加正态扰动

# 移动到GPU：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.cuda()
y = y.cuda()


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))

    def forward(self, x):
        return x @ self.w.t() + self.b


linear = LinearRegression()
linear.to(device)
print(next(linear.parameters()).is_cuda)

# 训练模型：
optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
loss_func = nn.MSELoss()


def train(epochs):
    tic = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = linear(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print({'epoch': epoch, 'loss': loss.item()})
    toc = time.time()
    print('time used: ', toc - tic)


train(500)

# 使用GPU：
