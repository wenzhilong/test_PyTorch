import torch
import torch.nn as nn
from torchkeras import summary


# 3、使用nn.Module基类构建模型并辅助应用模型容器进行封装：
# 使用nn.Sequential作为模型容器：
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1))
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.dense(x)
        return y


net = Net()
print(net)

summary(net, input_shape=(3, 32, 32))