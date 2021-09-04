import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
import argparse

from count_lipz import count_lipz

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--smooth_flag', type=bool, default=False, help="smooth_flag")
opt = parser.parse_args()

trainset_size = 10000
testset_size = 10000

length_fea = 100
batch_size = 100
num_epoch = 20

bias = 0.4
smooth_flag = opt.smooth_flag

def Makerand_fun(set_size, length_fea, bias=1):
    a = torch.randn(set_size, length_fea) + bias
    b = torch.randn(set_size, length_fea) - bias
    return torch.cat((a,b), dim=0)

def Metric_fun(features, center):
    return torch.abs(torch.mean(features, dim=1) - center).unsqueeze(1)

def Get_label(train_set, bias):
    # torch.cat((Metric_fun(train_set, bias), Metric_fun(train_set, -bias)), dim=1)
    return torch.softmax(torch.cat((Metric_fun(train_set, bias), Metric_fun(train_set, -bias)), dim=1), dim=1)

def Get_true_label(set_size):
    a = torch.zeros(set_size//2, 1)
    b = torch.ones(set_size//2, 1)
    return torch.cat((torch.cat((a,b), dim=1), 1-torch.cat((a,b), dim=1)), dim=0)

# Create Train Set
train_set = Makerand_fun(trainset_size//2, length_fea, bias).to(device)
train_label_0 = Get_label(train_set, bias)
# train_label_1 = Get_true_label(trainset_size).float().to(device)
train_label_1 = (train_label_0 > 0.5).float()

# print(train_label_0.mean().item())
# print(train_label_1.sum().item())

# Create Test Set
test_set = Makerand_fun(testset_size//2, length_fea, bias).to(device)
test_label_0 = Get_label(test_set, bias)
# test_label_1 = Get_true_label(testset_size).float().to(device)
test_label_1 = (test_label_0 > 0.5).float()

losses = []
acc_rates = []

net = nn.Sequential(
    nn.Linear(length_fea, length_fea),
    nn.Softplus(),
    nn.Linear(length_fea, length_fea),
    nn.Softplus(),
    nn.Linear(length_fea, length_fea),
    nn.Softplus(),
    nn.Linear(length_fea, 2),
).to(device)

# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MSELoss()
def criterion(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.8, weight_decay=1e-6)

# Train
net.train()
sample_size = trainset_size
for epoch in range(num_epoch):
    rand_index = random.sample(range(0, sample_size), sample_size)
    it_max = math.ceil(sample_size/batch_size)
    # for it in tqdm(range(it_max)):
    acc = 0
    for it in range(it_max):
        # pick a random example id 
        j = it * batch_size
        # select the corresponding example and label
        if j+batch_size <= sample_size:
            example = train_set[rand_index[j:j+batch_size], :]
            label_0 = train_label_0[rand_index[j:j+batch_size], :]
            label_1 = train_label_1[rand_index[j:j+batch_size], :]
        else:
            example = train_set[rand_index[j:], :]
            label_0 = train_label_0[rand_index[j:], :]
            label_1 = train_label_1[rand_index[j:], :]
        # do a forward pass on the example
        pred = net(example)
        pred_top_1 = torch.topk(pred, k=1, dim=1)[1]
        label_top_1 = torch.topk(label_1, k=1, dim=1)[1]
        acc += pred_top_1.eq(label_top_1.view_as(pred_top_1)).int().sum().item()
        # compute the loss according to your output and the label
        if smooth_flag:
            loss = criterion(pred, label_0)
        else :
            loss = criterion(pred, label_1)
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # gradient step
        optimizer.step()
        
    # End Epoch : append loss.item()
    acc_rate = acc/sample_size
    losses.append(loss.item())
    acc_rates.append(acc_rate)
    if epoch % 10 == 0 :
        print('Epoch:', epoch, ', loss:', loss.item(), ', acc:', acc_rate)
    # if epoch % 500 == 0 :
    #     torch.save(net.state_dict(), '%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch+epoch))


# Test
net.eval()
sample_size = testset_size
with torch.no_grad():
    it_max = math.ceil(sample_size/batch_size)
    acc = 0
    for it in range(it_max):
        # pick a random example id 
        j = it * batch_size
        # select the corresponding example and label
        if j+batch_size <= sample_size:
            example = test_set[j:j+batch_size, :]
            label_1 = test_label_1[j:j+batch_size, :]
        else:
            example = test_set[j:, :]
            label_1 = test_label_1[j:, :]
        
        # do a forward pass on the example
        pred = net(example)
        pred_top_1 = torch.topk(pred, k=1, dim=1)[1]
        label_top_1 = torch.topk(label_1, k=1, dim=1)[1]
        acc += pred_top_1.eq(label_top_1.view_as(pred_top_1)).int().sum().item()
    
    test_acc_rate = acc/sample_size

print('Test Acc: {}'.format(test_acc_rate))

Lipz = count_lipz(net, test_set, device, rand_times=16, eps=1e-1)
print(Lipz)