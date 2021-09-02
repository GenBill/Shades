import torch
import torch.nn as nn
import torch.nn.functional as F

def conti_CE(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

def shadow_train(args, model, dataloader, optimizer, scheduler, device):
    model.train()
    for epoch in (args.num_epoch):
        running_loss = 0
        running_acc = 0
        n_samples = 0
        for inputs_0, labels_0 in dataloader:
            batchSize = labels_0.size(0)
            n_samples += batchSize

            inputs_0 = inputs_0.to(device)
            labels_0 = labels_0.to(device)
            labels_0 = F.one_hot(labels_0, num_classes=args.num_class)

            for inputs_1, labels_1 in dataloader:
                break
            inputs_1 = inputs_1[:batchSize, :].to(device)
            labels_1 = labels_1[:batchSize, :].to(device)
            labels_1 = F.one_hot(labels_1, num_classes=args.num_class)

            # Create inputs & labels
            rand_vector = torch.rand((batchSize, 1), device=device)
            inputs = inputs_0 * rand_vector + inputs_1 * (1-rand_vector)
            labels = labels_0 * rand_vector + labels_1 * (1-rand_vector)
            del inputs_0, inputs_1, labels_0, labels_1, rand_vector

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = conti_CE(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * batchSize

            pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
            labels_top_1 = torch.topk(labels, k=1, dim=1)[1]
            running_acc += pred_top_1.eq(labels_top_1).int().sum().item()
        
        epoch_loss = running_loss / n_samples
        epoch_acc = running_acc / n_samples
        print('Epoch {}\nLoss : {:.6f}, Acc : {:.6f}'.format(epoch, epoch_loss, epoch_acc))
    
    return model
