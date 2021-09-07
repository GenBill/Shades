import torch

pred = torch.randn((16, 2))
print(pred)
values = pred.topk(k=2, dim=1)[0]
indices = pred.topk(k=2, dim=1)[1]
print(indices, indices.shape)
# 用max得到的结果，设置keepdim为True，避免降维。因为topk函数返回的index不降维，shape和输入一致。
_, indices_max = pred.max(dim=1, keepdim=True)

print(indices_max == indices)

loss = torch.mean(values[:,0] - values[:,1], dim=0)
print(loss)

m1 = torch.randn((16, 2), requires_grad=False)
m2 = torch.randn((16, 2), requires_grad=True)
m3 = m1*m2
loss = torch.mean(m3)
loss.backward()

loss.requires_grad
