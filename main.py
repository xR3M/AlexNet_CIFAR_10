import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchsummary
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# device configuration
device = torch.device('mps')

def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size = 0.1, shuffle = True):
    normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # transformation
    valid_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),normalize])
    if augment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])

# loading the dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size *num_train))


    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)


    return(train_loader, valid_loader)

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # defining transformation
    transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), normalize])

    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

# CIFAR10 dataset
train_loader, valid_loader = get_train_valid_loader(data_dir= './data', batch_size=64, augment=False, random_seed=1)
test_loader = get_test_loader(data_dir='./data', batch_size=64)

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.re2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

model = AlexNet(num_classes).to(device)

# Fine tuning: Freezing layers
for param in model.features[8:].parameters():
    param.requires_grad = False

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 0.005)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)


total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))



with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))






