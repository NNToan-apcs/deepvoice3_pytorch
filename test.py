import torchvision
import torch
use_cuda = torch.cuda.is_available()
print('torch.cuda.is_available()', torch.cuda.is_available())
train_dataset = torchvision.datasets.CIFAR10(root='cifar10_pytorch', download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, pin_memory=True)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
x, y = next(iter(train_dataloader))
print('x.device', x.device)
print('y.device', y.device)