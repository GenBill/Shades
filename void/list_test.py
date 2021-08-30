import torch
import torch.nn as nn

def identity_mapping():
    return nn.Sequential()

x = torch.rand(4,4)

id_layer = identity_mapping()
print(id_layer)

y = id_layer(x)
print(x-y)


nomat = torch.zeros(3,3)
a1 = torch.ones(3,3)
a2 = torch.ones(3,3) *4
nomat.unsqueeze_(2)
print(nomat)
nomat = torch.cat((nomat,a1.unsqueeze(2)), dim=2)
print(nomat)
nomat = torch.cat((nomat,a2.unsqueeze(2)), dim=2)
print(nomat)

stdmat = torch.std(nomat, dim=2)
loss = torch.mean(stdmat)
print(stdmat, loss)