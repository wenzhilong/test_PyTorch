import torch
import torch.nn as nn

# 查看GPU信息：
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == 'cuda':
    gpu_count = torch.cuda.device_count()
    print('gpu_count = ', gpu_count)


# 将张量在CPU和GPU之间移动：
print('='*20, '在CPU和GPU之间移动张量', '='*20)
tensor = torch.rand((100, 100))
tensor_gpu = tensor.to(device)  # 或者tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to('cpu')   # 或者tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu.device)

# 将模型中的全部张量移动到GPU上：
print('='*20, '将模型中的全部张量移动到GPU上', '='*20)
net = nn.Linear(2, 1)
print(next(net.parameters()).is_cuda)
net.to(device)
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)