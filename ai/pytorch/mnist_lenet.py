# Copyright 2015-2023 Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import argparse
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_mnist_dataset import LocalDataset

#print(torch.cuda.is_available())
#print(torch._C._cuda_getDeviceCount())
#print(torch.cuda.device_count())
print(os.getenv("CUDA_VISIBLE_DEVICES"))
device_count = len(os.getenv("CUDA_VISIBLE_DEVICES").split(",")) if os.getenv("CUDA_VISIBLE_DEVICES") else 0
print(f"cuda visible device count is:{device_count}")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batchsize', '-tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', '-sm', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    root = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/mnist.npz'

    if device_count > 1:
        print("=====================================")
        print("# train model with multi GPU")
        print('# minibatch-size: {0}'.format(args.batchsize))
        print('# epochs: {0}'.format(args.epochs))
        print("=====================================")
        device = 'cuda' if torch.cuda.is_available() else print("cuda is not available")
    elif device_count == 1:
        print("=====================================")
        print("# train model with a GPU")
        print('# minibatch-size: {0}'.format(args.batchsize))
        print('# epochs: {0}'.format(args.epochs))
        print("=====================================")
        device = torch.device('cuda' if torch.cuda.is_available()
                              else print("cuda is not available"))
    else:
        print("=====================================")
        print("# train model with CPU")
        print('# minibatch-size: {0}'.format(args.batchsize))
        print('# epochs: {0}'.format(args.epochs))
        print("=====================================")
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    train_data = LocalDataset(root, train=True)
    test_data = LocalDataset(root)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batchsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batchsize,
                                              shuffle=False)
    model = Net()
    if device_count > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), os.path.dirname(os.path.realpath(__file__)) + "/mnist_net.pt")


if __name__ == '__main__':
    main()
