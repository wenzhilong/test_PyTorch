import torch
import torch.nn as nn
from torchkeras import summary


# 3、使用nn.Module基类构建模型并辅助应用模型容器进行封装：
# 使用nn.ModuleDict作为模型容器：
# 注意：下面中的ModuleDict不能使用Python中的字典代替:
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers_dict = nn.ModuleDict({
            "conv1": nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            "pool": nn.MaxPool2d(kernel_size=2, stride=2),
            "conv2": nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            "dropout": nn.Dropout2d(p=0.1),
            "adaptive": nn.AdaptiveMaxPool2d((1, 1)),
            "flatten": nn.Flatten(),
            "linear1": nn.Linear(64, 32),
            "relu": nn.ReLU(),
            "linear2": nn.Linear(32, 1),
            "sigmoid": nn.Sigmoid()
            })

    def forward(self, x):
        layers = ["conv1", "pool", "conv2", "pool", "dropout", "adaptive",
                  "flatten", "linear1", "relu", "linear2", "sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x


net = Net()
print(net)

summary(net, input_shape=(3, 32, 32))