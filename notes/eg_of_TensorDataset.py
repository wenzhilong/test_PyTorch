import torch
from torch.utils.data import TensorDataset, DataLoader

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])

train_ids = TensorDataset(a, b)
# 输出：
for i in train_ids:
    print(i)  # tuple
# 切片输出：
print(train_ids[0:2])
print('-'*80)
for data, label in train_ids:
    print(data, label)

# 使用DataLoader对数据进行封装：
print('='*80)
train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
for i, data in enumerate(train_loader, 1):
    x_data, x_label = data
    print('batch:{0} \nx_data:\n{1} \nlabel:{2}'.format(i, x_data, x_label))
