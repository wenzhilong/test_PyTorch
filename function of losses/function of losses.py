import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1、使用内置损失函数
y_pred = torch.tensor([[10.0, 0.0, -10.0], [8.0, 8.0, 8.0]])
y_true = torch.tensor([0, 2])

# 直接调用交叉熵损失：
ce = nn.CrossEntropyLoss()(y_pred, y_true)
print(ce)

# 等价于先计算nn.LogSoftmax激活，再调用NLLLoss
y_pred_logsoftmax = nn.LogSoftmax(dim=1)(y_pred)
nll = nn.NLLLoss()(y_pred_logsoftmax, y_true)
print(nll)

# 2、 自定义损失函数：
