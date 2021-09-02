import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def identity_mapping(device):
    return nn.Sequential().to(device)

def teacher_creater(model_all, device):
    # model_ft
    return nn.Sequential(*(list(model_all.children())[:-1])).to(device)

def rota_creater(in_features, device):
    '''
    Linear !
    Question : Why not use Rotation Matrix , but Linear ? 
    Answer : Rotation Matrix cannot solve Chirality -> Linear
    '''
    return nn.Linear(in_features, in_features, bias=True).to(device)
    Se_rota = nn.Sequential(
        nn.LogSigmoid(),
        nn.Linear(in_features, in_features, bias=True),
    )
    return Se_rota.to(device)


def teacher_list_init(model_list, device):
    # init teacher_list
    teacher_list = []
    for i in model_list:
        teacher_list.append(teacher_creater(i,device))
    return teacher_list

def rota_list_init(teacher_list, in_features, device, quick_flag=False):
    # init rota_list
    model_num = len(teacher_list)
    rota_list = []
    if quick_flag:
        rota_list.append(identity_mapping(device))
        for i in range(model_num-1):
            rota_list.append(rota_creater(in_features, device))
    else:
        for i in range(model_num):
            rota_list.append(rota_creater(in_features, device))

    return rota_list


def rota_train(args, dataloader, model_list, device, quick_flag=False):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag
    '''
    model_num = len(model_list)
    in_features = model_list[0].fc.in_features
    teacher_list = teacher_list_init(model_list, device)
    rota_list = rota_list_init(teacher_list, in_features, device, quick_flag)
    
    rota_param = nn.Sequential(*rota_list)
    optimizer_rota = optim.SGD(
        rota_param.parameters(), nesterov=False, # True
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight, 
    )
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_rota, milestones=args.milestones, gamma=args.gamma)

    # init eval() & train()
    for i in range(model_num):
        teacher_list[i].eval()
        rota_list[i].train()
    
    print('Training the Rotavap ...')
    for epoch in range(args.num_epoch_0):
        running_loss = 0
        n_samples = 0
        for batch_num, (inputs, labels) in enumerate(dataloader):
            batchSize = inputs.size(0)
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_rota.zero_grad()
            with torch.no_grad():
                features = []
                for i in range(model_num):
                    features.append(nn.Flatten()(teacher_list[i](inputs)))
            
            outputs = rota_list[0](features[0]).unsqueeze(2)
            for i in range(1, model_num):
                # print(rota_list[i])
                # print(features[i].shape)
                temp = rota_list[i](features[i]).unsqueeze(2)
                outputs = torch.cat((outputs,temp), dim=2)
            
            # outputs = rota_list[0](teacher_list[0](inputs)).unsqueeze(2)
            # for i in range(1, model_num):
            #     temp = rota_list[i](teacher_list[i](inputs)).unsqueeze(2)
            #     outputs = torch.cat((outputs,temp), dim=2)
            
            loss = torch.mean(torch.std(outputs, dim=2))
            loss.backward()
            optimizer_rota.step()
            exp_lr_scheduler.step()

            running_loss += loss.item() * batchSize

        epoch_loss = running_loss / n_samples
        print('Epoch {}, Loss : {:.6f}'.format(epoch, epoch_loss))

    return teacher_list, rota_list



