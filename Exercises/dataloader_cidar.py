from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from cifar10_dataset import MyDataset
import numpy as np

if __name__ == "__main__":
    training_data = CIFAR10(root="data", train=True, transform=ToTensor())
    # training_data = MyDataset(root="data", train=True)
    # image, label = training_data.__getitem__(1234)
    training_dataloader = DataLoader(
        dataset=training_data,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    for images, labels in training_dataloader:
        print(images.shape)
        print(labels)
