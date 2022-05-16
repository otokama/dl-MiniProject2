'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_loss_hist = []
train_acc_hist = []
acc_hist = []

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernelsize=3, SkipConnectionKernelSize=1, dropout=False):
        super(BasicBlock, self).__init__()
        self.useDropout = dropout
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernelsize, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernelsize,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(0.3)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=SkipConnectionKernelSize, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.useDropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3Layer(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 AveragePKernelSize=4, Channel=(64,128,256),
                 ConvKernelSize=(3,3,3),
                 SkipKernelSize=(1,1,1), dropout=False):
        super(ResNet3Layer, self).__init__()
        self.in_planes = Channel[0]
        self.conv1 = nn.Conv2d(3, Channel[0], kernel_size=ConvKernelSize[0],
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Channel[0])
        self.layer1 = self._make_layer(block, Channel[0], num_blocks[0], 1, ConvKernelSize[0], SkipKernelSize[0], dropout)
        self.layer2 = self._make_layer(block, Channel[1], num_blocks[1], 2, ConvKernelSize[1], SkipKernelSize[1], dropout)
        self.layer3 = self._make_layer(block, Channel[2], num_blocks[2], 2, ConvKernelSize[2], SkipKernelSize[2], dropout)
        self.avgpool = nn.AvgPool2d(AveragePKernelSize, stride=1)
        self.linear = nn.Linear(Channel[2]*((8-AveragePKernelSize+1)**2)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ConvKSize, SkipKSize, dropout):
        if num_blocks == 0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ConvKSize, SkipKSize, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # (32, 32) = (32, 32)
        out = self.layer1(out)  # (32, 32) = (32, 32)
        out = self.layer2(out)  # (16, 16) = (32, 32)
        out = self.layer3(out)  # (8, 8) = (16, 16)
        out = self.avgpool(out)  # 64, 512, (1, 1) = (8, 8)  # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # (64, 512) = (64, 512, 1, 1)
        out = self.linear(out)  # (64, 10) = (64, 512)
        return out

def BuildBasicModelWithParameter(Bi, Ci, Fi, Ki, P, dropout, num_classes):
    return ResNet3Layer(BasicBlock,
                    num_blocks=Bi,
                    Channel=Ci,
                    ConvKernelSize=Fi,
                    SkipKernelSize=Ki,
                    AveragePKernelSize=P, dropout=dropout, num_classes=num_classes)


def BuildNet():
    global model_name
    model_name = ''
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    if args.model == 0: # ResNet-20
        model_name = 'ResNet-20'
        return BuildBasicModelWithParameter((3,3,3), (16,32,64), (3,3,3), (1,1,1), 8, False, num_classes)
    elif args.model == 1: # ResNet-32
        model_name = 'ResNet-32'
        return BuildBasicModelWithParameter((5,5,5), (16,32,64), (3,3,3), (1,1,1), 8, False, num_classes)
    elif args.model == 2: # ResNet-56
        model_name = 'ResNet-56'
        return BuildBasicModelWithParameter((9,9,9), (16,32,64), (3,3,3), (1,1,1), 8, False, num_classes)
    elif args.model == 3: # ResNet-110
        model_name = 'ResNet-110'
        return BuildBasicModelWithParameter((36,36,36), (16,32,64), (3,3,3), (1,1,1), 8, False, num_classes)
    else:
        raise Exception("No such ResNet model")



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint', default=50, type=int, help='every # epoches to save')
    parser.add_argument('--epoch', default=200, type=int, help='train epoch # total for one time')
    parser.add_argument('--model', default=0, type=int, help='20 (for Resnet20)')
    parser.add_argument('--dataset',default='cifar10', help='dataset to use (cifar10 or cifar100)')
    args = parser.parse_args()
    return args

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss_hist.append(train_loss/len(trainloader))            
    train_acc_hist.append(100 * correct / total)         
    print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    # Save checkpoint.
    acc = 100.*correct/total
    acc_hist.append(acc)
    if acc > best_acc:
        if epoch % args.checkpoint == 0:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
        if not os.path.isdir('best_model'):
                os.mkdir('best_model')
        torch.save(net.state_dict(), './best_model/{}_{}.pt'.format(model_name, args.dataset))    
        best_acc = acc
    print("Best Accuracy: %.2f%%"%best_acc)

if __name__ == '__main__':
    global args
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Training on', device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        raise Exception("No such dataset")
        exit()

    # Model
    print('==> Building model Resnet ', args.model)
    net = BuildNet()
    print("\n", net, "\n")
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print("Count parameters: ", count_parameters(net))

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=True,
                          weight_decay=0.0001 
                          )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

    # Saving figures
    plt.figure(figsize=(5,3), dpi=200)
    plt.rcParams["font.family"] = "Times New Roman"
    x = range(1,args.epoch+1)
    plt.plot(x, train_loss_hist, linestyle='-.',  color='r',antialiased=True, lw=2, label='Train Loss')
    plt.plot(x, np.array(train_acc_hist)/100, linestyle='dashed',  color='g',antialiased=True, lw=2, label='Train Acc')
    plt.plot(x, np.array(acc_hist)/100, linestyle='-',  color='b',antialiased=True, lw=2, label='Test Acc')
    plt.legend(loc=1)
    plt.xlabel('Epoch', size=9)
    plt.grid(axis='x')
    plt.grid(axis='y')
    if not os.path.isdir('plots'):
                os.mkdir('plots')
    plt.savefig('./plots/{}_{}_loss_acc.png'.format(model_name, args.dataset))
    plt.savefig('./plots/{}_{}_loss_acc.svg'.format(model_name, args.dataset))
