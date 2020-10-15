import torch
import torch.nn as nn
import torchkeras
from torchkeras import summary, Model
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# 准备数据：
transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root='./data/minist/', train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root='./data/minist/', train=False, download=True, transform=transform)

dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = DataLoader(ds_valid, batch_size=128, shuffle=True, num_workers=4)
# print(len(ds_train))
# print(len(ds_valid))

# plt.figure(figsize=(8, 8))
# for i in range(9):
#     img, label = ds_train[i]
#     img = torch.squeeze(img)
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title('label = %d' % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()


# 使用类风格进行训练：
class CNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


model = torchkeras.Model(CNNModule())
# print(model)
#
# summary(model, input_shape=(1, 32, 32))


# 训练模型：
def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    # return accuracy_score(y_true.numpy(), y_pred_cls.numpy()) CPU版代码
    return accuracy_score(y_true.cpu().numpy(), y_pred_cls.cpu().numpy())  # 先将数据移动到CPU上，然后才能转换成numpy数组


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model.compile(loss_func=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
              metrics_dict={'accuracy': accuracy}, device=device)  # GPU版代码添加了device


# 评估模型：
def plot_metric(dfhistory, metirc):
    train_metrics = dfhistory[metirc]
    val_metrics = dfhistory['val_'+metirc]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('training and validation ' + metirc)
    plt.xlabel('epochs')
    plt.ylabel(metirc)
    plt.legend(['train_' + metirc, 'val_' + metirc])
    plt.show()


if __name__ == '__main__':
    df_history = model.fit(3, dl_train=dl_train, dl_val=dl_valid, log_step_freq=100)
    plot_metric(df_history, 'loss')
    plot_metric(df_history, 'accuracy')
    print(model.evaluate(dl_valid))
    print(model.predict(dl_valid)[0: 10])
    # 保存模型：
    torch.save(model.state_dict(), 'model_parameter.pkl')
    model_clone = torchkeras.Model(CNNModule())
    model_clone.load_state_dict(torch.load('model_parameter.pkl'))
    model_clone.compile(loss_func=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
                        metrics_dict={'accuracy': accuracy}, device=device)
    print(model_clone.evaluate(dl_valid))