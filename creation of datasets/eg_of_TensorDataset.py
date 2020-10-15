import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from sklearn import datasets

# 根据Tensor创建数据集：
iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data), torch.tensor(iris.target))

# 分割成训练集和预测集
n_train = int(len(ds_iris)*0.8)
n_valid = len(ds_iris) - n_train
ds_train, ds_valid = random_split(ds_iris, [n_train, n_valid])

print(type(ds_iris))
print(type(ds_train))

# 使用DataLoader加载数据集：
dl_train, dl_valid = DataLoader(ds_train, batch_size=8), DataLoader(ds_valid, batch_size=8)
for features, labels in dl_train:
    print(features)
    print(labels)
    break

# 演示加法运算符（`+`）的合并作用

ds_data = ds_train + ds_valid

print('len(ds_train) = ',len(ds_train))
print('len(ds_valid) = ',len(ds_valid))
print('len(ds_train+ds_valid) = ',len(ds_data))

print(type(ds_data))