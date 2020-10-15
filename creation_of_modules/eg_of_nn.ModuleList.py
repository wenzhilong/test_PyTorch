import torch
import torch.nn as nn
from torchkeras import summary


# 3、使用nn.Module基类构建模型并辅助应用模型容器进行封装：
# 使用nn.ModuleList作为模型容器：
# 注意：下面中的ModuleList不能使用Python中的列表代替：
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()]
        )

    def forward(self, x):
        for layer in self.layers:
            # print(layer)  # layer是layers中的各行
            x = layer(x)
        return x


net = Net()
print(net)

summary(net, input_shape=(3, 32, 32))
