import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from pprint import pprint

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        all_bboxes = []
        all_labels = []
        for obj in data["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            all_bboxes.append([xmin, ymin, xmax, ymax])
            all_labels.append(self.categories.index(obj["name"]))
        all_bboxes = torch.FloatTensor(all_bboxes)
        all_labels = torch.LongTensor(all_labels)
        target = {
            "boxes": all_bboxes,
            "labels": all_labels
        }
        return image, target

if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="my_pascal_voc", year="2012", image_set="train", download=False, transform=transform)
    image, target = dataset[2000]
    print(image.shape)
    print(target)


