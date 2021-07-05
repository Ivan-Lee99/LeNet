

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import struct


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        sum_loss = 0.0
        for index, data in enumerate(train_loader):
            input, label = data
            # print(input.shape)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # forward + back + optimize
            output = model(input)
            loss = criterion(output, label)
            print(loss)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if index % 100 == 99:
                print('[%d,%d] loss:%.03f' % (epoch + 1, index + 1, sum_loss / 100))
                sum_loss = 0.0
    # torch.save(model, './LeNet')


def test(test_loader, model):
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        # torch.max并不是np.max一个意思，是用以计算sofamax的分类类别的，建议CSDN查一下
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    print("+++++++++++++", total, correct)


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


batch_size = 64
lr = 0.001
epochs = 10

net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

train_image, train_label = load_mnist("./mnist_dataset")
print(train_image.shape, type(train_image), train_label.shape, type(train_label))
x_data = torch.from_numpy(train_image.reshape(-1, 1, 28, 28)).float()
y_data = torch.from_numpy(train_label).long()
dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

test_image, test_label = load_mnist("./mnist_dataset", "t10k")
a_data = torch.from_numpy(test_image.reshape(-1, 1, 28, 28)).float()
b_data = torch.from_numpy(test_label).long()
dataset = TensorDataset(a_data, b_data)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

"""for index, data in enumerate(train_loader):
    input, label = data
    train_image = train_image.reshape(-1, 1, 28, 28)
    print(train_image.shape)
    cv2.imshow("number", train_image[0][0])
    cv2.waitKey(0)"""
"""
print(train_loader)
for i in range(5):
    for index, data in enumerate(train_loader):
        input, label = data
        print(input.shape, label.shape)
"""
train(net, criterion, optimizer, epochs)
test(test_loader, net)

