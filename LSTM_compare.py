# coding: utf-8
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import csv

EPOCH = 8              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
TIME_STEP = 4         # rnn time step / image height
INPUT_SIZE = 4         # rnn input size / image width
LR = 0.0005               # learning rate
#DOWNLOAD_MNIST = False   # set to True if haven't download the data

csvfile = open('train_bad.csv')
reader = csv.reader(csvfile)
labels = []
feature = []
for line in reader:
    tmpline = int(line[11])
    temfea = torch.ones(4,4)
    temfea[0][0] = float((float(line[0])-101)/702)
    temfea[0][1] = float((float(line[1])-5)/604)
    temfea[0][2] = float((float(line[2])-1)/7)
    temfea[0][3] = float((float(line[3])-1)/11)
    temfea[1][0] = float((float(line[4])-5)/575)
    temfea[1][1] = float((float(line[5])-101)/702)
    temfea[1][2] = float((float(line[6])-1)/7)
    temfea[1][3] = float((float(line[7])-1)/11)
    temfea[2][0] = float((float(line[8])-2)/1)
    temfea[2][1] = float((float(line[9])-0)/7)
    temfea[2][2] = float((float(line[10])-1)/4)
    temfea[2][3] = float(0)
    temfea[3][0] = float(0)
    temfea[3][1] = float(0)
    temfea[3][2] = float(0)
    temfea[3][3] = float(0)
    labels.append(tmpline)
    feature.append(temfea)
csvfile.close()

#tlabels = torch.from_numpy(labels)
#tfeature = torch.from_numpy(feature)


#print(labels[1378])
#print(feature[1378])
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

#print(dataset)

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(MyDataset(feature, labels), batch_size=BATCH_SIZE, shuffle=True)
train_loader = torch.utils.data.DataLoader(MyDataset(feature, labels), batch_size=BATCH_SIZE, shuffle=True)

"""
i = 1
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        if i<2:
            print(b_x[2])
            #print(b_x.view(-1, 11, 1))
            i = i+1

"""
csvfile = open('test_last2w.csv')
reader1 = csv.reader(csvfile)
test_x = []
test_y = []
for line in reader1:
    cstar = torch.ones(1,1)
    cstar = int(line[11])
    csfea = torch.ones(4,4)
    csfea[0][0] = float((float(line[0])-101)/702)
    csfea[0][1] = float((float(line[1])-5)/604)
    csfea[0][2] = float((float(line[2])-1)/7)
    csfea[0][3] = float((float(line[3])-1)/11)
    csfea[1][0] = float((float(line[4])-5)/575)
    csfea[1][1] = float((float(line[5])-101)/702)
    csfea[1][2] = float((float(line[6])-1)/7)
    csfea[1][3] = float((float(line[7])-1)/11)
    csfea[2][0] = float((float(line[8])-2)/1)
    csfea[2][1] = float((float(line[9])-0)/7)
    csfea[2][2] = float((float(line[10])-1)/4)
    csfea[2][3] = float(0)
    csfea[3][0] = float(0)
    csfea[3][1] = float(0)
    csfea[3][2] = float(0)
    csfea[3][3] = float(0)
    test_y.append(cstar)
    test_x.append(csfea)
csvfile.close()

testdata = MyDataset(test_x,test_y)
#test_x = test_x.type(torch.FloatTensor)
#test_y = test_y.numpy().squeeze()  # covert to numpy array

#test_x = Variable(test_x,requires_grad = True)

test_loader = torch.utils.data.DataLoader(MyDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=True)



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=50,         # rnn hidden unit
            num_layers=8,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(50, 9)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_c,h_n) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = F.relu(self.out(r_out[:, -1, :]))
        return out


rnn = RNN()
print(rnn)

#optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
optimizer = torch.optim.RMSprop(rnn.parameters(), lr=LR, alpha=0.9)
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

bzd = 0
zhenbzd =0
zhun = 0
zhenzhun = 0
quan = 0
zhenquan = 0
dui = 0 
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 4, 4)              # reshape x to (batch, time_step, input_size)
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()
        if step%100 == 0:
            for datala in test_loader:
                tez,leibiao = datala
                tez = tez.squeeze(1)
                tez = Variable(tez, volatile=True)
                leibiao = Variable(leibiao, volatile=True)
                test_output = rnn(tez)  
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                tloss = loss_func(output,leibiao)
                for k in range(50):
                    if int(pred_y[k]) == int(leibiao[k]):
                        dui = dui+1
                        #print(pred_y[k])
                        #print(leibiao[k])
                    if int(pred_y[k]) != 1:                 #模型觉得是异常的
                        zhun = zhun+1                  
                        if int(leibiao[k]) != 1:            #确实是异常的
                            zhenzhun = zhenzhun+1
                    if int(leibiao[k]) != 1:                #本就是是异常的
                        quan = quan+1
                        if int(pred_y[k]) != 1:             #模型检测出来是异常的
                            zhenquan = zhenquan+1
                    if int(pred_y[k])!=1 and int(leibiao[k])!=1:
                        bzd=bzd+1
                        if int(pred_y[k])==int(leibiao[k]):
                            zhenbzd=zhenbzd+1
                k = 0
            accuracy = float(dui)/float(4000)
            if zhun==0:
                precision=0
            else:
                precision = float(zhenzhun)/float(zhun)
            if quan==0:
                recall=0
            else:
                recall = float(zhenquan)/float(quan)
            if (precision+recall) == 0:
                score = 0
            else:
                score = 2*precision*recall/(precision+recall)
            if bzd==0:
                RSRAA=0
            else:
                RSRAA=float(zhenbzd)/float(bzd)
            dui = 0
            zhun = 0
            zhenzhun = 0
            quan = 0
            zhenquan = 0
            bzd = 0
            zhenbzd = 0
            print('Epoch: ', epoch)
            print('ACC:',accuracy)
            print('Precision:',precision)
            print('Recall:',recall)
            print('F1 measure:',score)
            print('RSRAA:',RSRAA)
            print('train loss: %.4f' % loss.data.numpy())
            print('test loss: %.4f' % tloss.data.numpy())
            #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


                                # apply gradients
"""
        if step % 50 == 0:
            o = 0
            
            test_output = rnn(test_x)
            print(o)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            for j in range(2000):
                if pred_y[j]==test_y[j]:
                    o = o+1
            print(float(o/2000))  
            
            

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 11, 1))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

"""


print("ooo")


















