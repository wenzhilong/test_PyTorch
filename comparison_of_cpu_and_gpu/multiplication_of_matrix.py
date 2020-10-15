import time
import torch
from torch import nn

# 1、矩阵乘法范例：
# 使用CPU：
a = torch.rand((10000, 2000))
b = torch.rand((2000, 10000))
tic = time.time()
c = torch.matmul(a, b)
toc = time.time()


print(toc - tic)
print(a.device)
print(b.device)


# 使用GPU：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand((10000, 2000), device=device)  # 可以在指定GPU上创建张量
y = torch.rand((2000, 10000)).to(device)  # 也可以在CPU上创建张量之后移动到GPU
tic = time.time()
z = torch.matmul(x, y)
toc = time.time()
print(toc - tic)
print(x.device)
print(y.device)
