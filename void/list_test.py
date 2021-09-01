import torch
import torch.nn as nn
import torchvision

def identity_mapping():
    return nn.Sequential()

x = torch.rand(4,4)

id_layer = identity_mapping()
print(id_layer)

y = id_layer(x)
print(x-y)


nomat = torch.zeros(3,3)
a1 = torch.ones(3,3)
a2 = torch.ones(3,3)
nomat.unsqueeze_(2)
print(nomat)
nomat = torch.cat((nomat,a1.unsqueeze(2)), dim=2)
print(nomat)
nomat = torch.cat((nomat,a2.unsqueeze(2)), dim=2)
print(nomat)

stdmat = torch.std(nomat, dim=2)
loss = torch.mean(stdmat)
print(stdmat, loss)


fea_0 = torch.ones(3,3)
fea_1 = torch.ones(3,3)

klloss = nn.KLDivLoss(reduction='batchmean')(fea_0, fea_1)
print('batchmean', klloss)

klloss = nn.KLDivLoss(reduction='mean')(fea_0, fea_1)
print('mean', klloss)

klloss = nn.KLDivLoss(reduction='none')(fea_0, fea_1)
print('none', klloss)


b1 = torch.ones(3,1)
b2 = torch.ones(3,1,1)

b3 = nn.Flatten()(b2)
print(b2.shape, b3.shape)


student = torchvision.models.resnet18(pretrained=False)
student_ft = nn.Sequential(*(list(student.children())[:-1]))
student_fc = nn.Sequential(*(list(student.children())[-1:]))

print(student_ft, student_fc)

z1 = torch.ones(3,3)
z0 = z1[0:1, :]
print(z0.shape)