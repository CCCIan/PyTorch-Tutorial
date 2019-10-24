﻿# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:46:33 2019

@author: CIan
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import csv
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import argparse
import torch.optim as optim


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 50)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 8)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=10,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()



csvfile = open('train_bad.csv')
reader = csv.reader(csvfile)
labels = []
feature = []
for line in reader:
    tmpline = int(line[11])
    temfea = torch.ones(1,11)
    temfea[0][0] = float((float(line[0])-101)/702)
    temfea[0][1] = float((float(line[1])-5)/604)
    temfea[0][2] = float((float(line[2])-1)/7)
    temfea[0][3] = float((float(line[3])-1)/11)
    temfea[0][4] = float((float(line[4])-5)/575)
    temfea[0][5] = float((float(line[5])-101)/702)
    temfea[0][6] = float((float(line[6])-1)/7)
    temfea[0][7] = float((float(line[7])-1)/11)
    temfea[0][8] = float((float(line[8])-2)/1)
    temfea[0][9] = float((float(line[9])-0)/7)
    temfea[0][10] = float((float(line[10])-1)/4)
    labels.append(tmpline)
    feature.append(temfea)
csvfile.close()

#tlabels = torch.from_numpy(labels)
#tfeature = torch.from_numpy(feature)



class MyDataset(data.Dataset):
    def __init__(self, feature, labels):
        self.feature = feature
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img, tar = self.feature[index], self.labels[index]
        return img, tar

    def __len__(self):
        return len(self.feature)

dataset = MyDataset(feature, labels)


csvfile = open('test_last2w.csv')
reader1 = csv.reader(csvfile)
test_x = []
test_y = []
for line in reader1:
    cstar = torch.ones(1,1)
    cstar = int(line[11])
    csfea = torch.ones(1,11)
    csfea[0][0] = float((float(line[0])-101)/702)
    csfea[0][1] = float((float(line[1])-5)/604)
    csfea[0][2] = float((float(line[2])-1)/7)
    csfea[0][3] = float((float(line[3])-1)/11)
    csfea[0][4] = float((float(line[4])-5)/575)
    csfea[0][5] = float((float(line[5])-101)/702)
    csfea[0][6] = float((float(line[6])-1)/7)
    csfea[0][7] = float((float(line[7])-1)/11)
    csfea[0][8] = float((float(line[8])-2)/1)
    csfea[0][9] = float((float(line[9])-0)/7)
    csfea[0][10] = float((float(line[10])-1)/4)
    test_y.append(cstar)
    test_x.append(csfea)
csvfile.close()

testdata = MyDataset(test_x,test_y)


test_loader = torch.utils.data.DataLoader(MyDataset(test_x, test_y), batch_size=args.batch_size, shuffle=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=7, dropout=0.05)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


batch_size = args.batch_size
n_classes = 9
input_channels = 1
seq_length = int(11 / input_channels)
epochs = args.epochs
steps = 0

print(args)


permute = torch.Tensor(np.random.permutation(9).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)



lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr