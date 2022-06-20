import os
from fnmatch import translate

import numpy as np
import pandas as pd
from PIL import Image
from time import time
from matplotlib import pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision import models


def main():
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
            self.linear1 = nn.Linear(9216, 256)
            self.linear2 = nn.Linear(256, 15)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.drop1 = nn.Dropout(0.25)
            self.drop2 = nn.Dropout(0.5)

            self.myModel = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(9216, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 15)
            ).to(device)

        # Sequential forward function
        """
        def forward(self, x):
            x = self.myModel(x)
            return x
        """

        # Residual connection
        def forward(self, input):
            output = self.conv1(input)
            output += self.relu(output)
            output += self.pool(output)

            output = self.conv2(output)
            output = self.relu(output)
            output = self.pool(output)

            output = self.conv3(output)
            output = self.relu(output)
            output = self.pool(output)

            output = self.conv4(output)
            output = self.relu(output)
            output = self.pool(output)

            output = self.conv5(output)
            output = self.relu(output)
            output = self.pool(output)

            output = self.flatten(output)
            output = self.linear1(output)
            output = self.relu(output)

            output = self.linear2(output)
            return output

    # Train function
    def Train(epoch, print_every=1):
        total_loss = 0
        start_time = time()

        accuracy = []

        for i, batch in enumerate(train_dataloader, 1):
            minput = batch[0].to(device)  # Get batch of images from our train dataloader
            target = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas

            moutput = model(minput)  # output by our model

            loss = criterion(moutput, target)  # compute cross entropy loss
            total_loss += loss.item()

            optimizer.zero_grad()  # Clear the gradients if exists. (Gradients are used for back-propogation.)
            loss.backward()  # Back propogate the losses
            optimizer.step()  # Update Model parameters

            argmax = moutput.argmax(dim=1)  # Get the class index with maximum probability predicted by the model
            accuracy.append(
                (target == argmax).sum().item() / target.shape[0])  # calculate accuracy by comparing to target tensor

            if i % print_every == 0:
                print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
                    epoch, i, len(train_dataloader), loss.item(), sum(accuracy) / len(accuracy), time() - start_time
                ))

        return total_loss / len(train_dataloader)  # Returning Average Training Loss

    # Test function
    def Test(epoch):
        total_loss = 0
        start_time = time()

        accuracy = []

        with torch.no_grad():  # disable calculations of gradients for all pytorch operations inside the block
            for i, batch in enumerate(test_dataloader):
                minput = batch[0].to(device)  # Get batch of images from our test dataloader
                target = batch[1].to(
                    device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas
                moutput = model(minput)  # output by our model

                loss = criterion(moutput, target)  # compute cross entropy loss
                total_loss += loss.item()

                # To get the probabilities for different classes we need to apply a softmax operation on moutput
                argmax = moutput.argmax(
                    dim=1)  # Find the index(0, 1 or 2) with maximum score (which denotes class with maximum probability)
                accuracy.append((target == argmax).sum().item() / target.shape[
                    0])  # Find the accuracy of the batch by comparing it with actual targets

        print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
            epoch, total_loss / len(test_dataloader), sum(accuracy) / len(accuracy), time() - start_time
        ))
        return total_loss / len(test_dataloader)  # Returning Average Testing Loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using:', device)

    dataset_path = 'Dataset'

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=40),

        transforms.Resize(300),
        transforms.CenterCrop(256),

        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (2989, 1494))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Create a Test DataLoader using Test Dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    figsize = (16, 16)

    model = MyModel().to(device)

    lr = 0.001

    model = MyModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.layer4.requires_grad = True

    model.to(device)

    model.to(device)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)

    summary(model, (3, 256, 256))

    model.to(device)

    Test(0)

    train_loss = []
    test_loss = []

    for epoch in range(1, 31):
        train_loss.append(Train(epoch, 1))
        test_loss.append(Test(epoch))

        print('\n')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'model' + str(epoch) + '.pth')

    # Running test again
    """
    model.load_state_dict(torch.load("model_30.pth"))
    model.eval()
    model.train()
    for epoch in range(31, 51):
        train_loss.append(Train(epoch, 1))
        test_loss.append(Test(epoch))

        print('\n')

        if epoch % 10 == 0:
            torch.save(model, 'model_' + str(epoch) + '.pth')
    """
    return True


if __name__ == '__main__':
    main()
