import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchkeras

# 正负样本数量：
n_positive, n_negative = 200, 6000

# 生成正样本，小圆环分布：
r_p = 5.0 + torch.normal(0.0, 1.0, size=[n_positive, 1])
theta_p = 2 * np.pi * torch.rand([n_positive, 1])
Xp = torch.cat([r_p*torch.cos(theta_p), r_p*torch.sin(theta_p)], axis=1)
Yp = torch.ones_like(r_p)

# 生成负样本，大圆环分布：
r_n = 8.0 + torch.normal(0.0, 1.0, size=[n_negative, 1])
theta_n = 2 * np.pi * torch.rand([n_negative, 1])
Xn = torch.cat([r_n*torch.cos(theta_n), r_n*torch.sin(theta_n)], axis=1)
Yn = torch.zeros_like(r_n)

# 汇总样本：
X = torch.cat([Xp, Xn], axis=0)
Y = torch.cat([Yp, Yn], axis=0)

# 可视化：
plt.figure(figsize=(6, 6))
plt.scatter(Xp[:, 0], Xp[:, 1], c='r')
plt.scatter(Xn[:, 0], Xn[:, 1], c='g')
plt.legend(['positive', 'negative'])
# plt.show()

# 分割训练集、验证集：
ds = TensorDataset(X, Y)
ds_train, ds_valid = random_split(ds, [int(len(ds)*0.7), len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=2)
dl_valid = DataLoader(ds_valid, batch_size=100, num_workers=2)


# 定义模型：
class DNNModel(torchkeras.Model):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y


net = DNNModel()
# net.summary(input_shape=(2, ))
# print(net)


# 自定义损失函数：
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        bce = torch.nn.BCELoss(reduction='none')(y_pred, y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)

        return loss


# 训练模型：
# 准确率：
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                         torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc


# l2正则化：
def l2loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:  # 一般不对称偏置项使用正则
            l2_loss = l2_loss + (0.5*alpha*torch.sum(torch.pow(param, 2)))

    return l2_loss


# L1正则化：
def l1loss(model, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))

    return l1_loss


# 将L2和L1正则添加到FocalLoss损失，一起作为目标函数：
def focal_loss_with_regularization(y_pred, y_true):
    focal = FocalLoss()(y_pred, y_true)
    l2_loss = l2loss(net, 0.001)  # 注意设置正则化项系数
    l1_loss = l1loss(net, 0.001)
    total_loss = focal + l1_loss + l2_loss

    return total_loss


net.compile(loss_func=focal_loss_with_regularization, optimizer=torch.optim.Adam(net.parameters(), lr=0.01),
            metrics_dict={'accuracy': accuracy})

if __name__ == '__main__':
    df_history = net.fit(30, dl_train=dl_train, dl_val=dl_valid, log_step_freq=30)
    # 结果可视化


