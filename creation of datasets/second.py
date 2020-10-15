import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

# 根据图片目录创建图片数据集：
img = Image.open('../snow mountain.jpg')
img.show()

# 随机数值翻转：
img = transforms.RandomVerticalFlip()(img)
img.show()

# 随机旋转：
img = transforms.RandomRotation(45)(img)
img.show()

