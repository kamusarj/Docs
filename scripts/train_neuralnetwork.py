import torch.optim
from torchvision.transforms import ToTensor
from dataset import CIFARDataset
from models import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report



if __name__ == '__main__':
    num_epochs = 100
    train_dataset = CIFARDataset(root="./data", train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = CIFARDataset(root="./data", train=False)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = SimpleNeuralNetwork(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            # if iter + 1 % 10:
            #     print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        print("Epoch {}".format(epoch+1))
        print(classification_report(all_labels, all_predictions))




