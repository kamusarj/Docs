import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class AnimalDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            pass
        data_files = []
        data_files_path = []
        self.images_path = []
        self.images = []
        self.labels = []
        self.categories = os.listdir(root)
        for i, category in enumerate(self.categories):
            data_files_path = os.path.join(self.root, category)
            for file_name in os.listdir(data_files_path):
                file_path = os.path.join(data_files_path, file_name)
                self.images_path.append(file_path)
                self.labels.append(i)
                # self.images.append(Image.open(file_path))
                self.images.append(cv2.imread(file_path))

    def __len__(self):
        return len(self.images)

    def __getCategories__(self):
        return self.categories

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


if __name__ == "__main__":
    root = "./Animals-10"
    dataset = AnimalDataset(root=root, train=True)
    # print(len(dataset))
    # image, label = dataset.__getitem__(1300)
    # print(image.shape)
    # # print(image.show())
    # print(label)
    # print(type(image))
    # print(image.dtype)
    # print(dataset.__getCategories__())
    # # image = cv2.resize(image, (320, 320))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    image, label = dataset.__getitem__(1300)
    print(label)
    cv2.imshow("image", image)
    cv2.waitKey(0)
