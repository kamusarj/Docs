import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from tqdm.autonotebook import tqdm
import argparse
import os
import numpy as np



class VOCDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform=None, target_transform=None):
        super().__init__(root, year, image_set, download, transform, target_transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']


    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.array(target, np.int64)
        target[target == 255] = 0
        return image, target

def get_args():
    parser = argparse.ArgumentParser(description="Train DeeplabV3 model")
    parser.add_argument("--data_path", "-d", type=str, default="my_pascal_voc", help="Path to dataset")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard", help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models",
                        help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = Resize((args.image_size, args.image_size))
    train_dataset = VOCDataset(root="my_pascal_voc", year="2012", image_set="train", download=False, transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False
    )
    test_dataset = VOCDataset(root="my_pascal_voc", year="2012", image_set="val", download=False,
                               transform=transform, target_transform=target_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        drop_last=False
    )
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    writer = SummaryWriter(args.log_folder)
    num_iters_per_epoch = len(train_dataloader)
    acc_metric = MulticlassAccuracy(num_classes=len(train_dataset.classes)).to(device)
    mIOU_metric = MulticlassJaccardIndex(num_classes=len(train_dataset.classes)).to(device)
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        all_losses = []
        for iter, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            result = model(images)
            output = result["out"]
            loss = criterion(output, targets)
            all_losses.append(loss.item())
            avg_loss = np.mean(all_losses)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch+1, args.num_epochs, avg_loss))
            writer.add_scalar("Train/Loss", avg_loss, epoch*num_iters_per_epoch+iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        progress_bar = tqdm(test_dataloader, colour="cyan")
        test_acc = []
        test_mIOU = []
        with torch.no_grad():  # torch.inference_mode()
            for images, targets in progress_bar:
                images = images.to(device)
                targets = targets.to(device)
                result = model(images)
                output = result["out"]  # B, C, H, W  (pred).  B, H, W (gt)
                test_acc.append(acc_metric(output, targets).item())
                test_mIOU.append(mIOU_metric(output, targets).item())
        avg_acc = np.mean(test_acc)
        avg_mIOU = np.mean(test_mIOU)
        print("Accuracy: {:0.4f}. mIOU: {:0.4f}".format(avg_acc, avg_mIOU))
        writer.add_scalar("Test/Accuracy", avg_acc, epoch)
        writer.add_scalar("Test/mIOU", avg_mIOU, epoch)





if __name__ == '__main__':
    args = get_args()
    train(args)
