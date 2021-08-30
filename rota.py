import torch
import torch.nn as nn

def identity_mapping(device):
    return nn.Sequential().to(device)

def teacher_creater(model_all, device):
    # model_ft
    return nn.Sequential(*(list(model_all.children())[:-1])).to(device)

def rota_creater(in_features, device):
    '''
    Linear without bias
    Question : Why not use Rotation Matrix , but Linear without bias ? 
    Answer : Rotation Matrix cannot solve Chirality -> Linear
    '''
    return nn.Linear(in_features, in_features, bias=False, device=device)


def teacher_list_init(model_list, device):
    # init teacher_list
    teacher_list = []
    for i in model_list:
        teacher_list.append(teacher_creater(i,device))
    return teacher_list

def rota_list_init(teacher_list, device, quick_flag=False):
    # init rota_list
    model_num = len(teacher_list)
    rota_list = []
    if quick_flag:
        rota_list.append(identity_mapping(device))
        for i in range(model_num-1):
            in_features = teacher_list[i][-1].out_features
            rota_list.append(rota_creater(in_features, device))
    else:
        for i in range(model_num):
            in_features = teacher_list[i][-1].out_features
            rota_list.append(rota_creater(in_features, device))

    return rota_list


def rota_train(dataloader, optimizer, num_epoch, model_list, device, quick_flag=False):
    model_num = len(model_list)
    teacher_list = teacher_list_init(model_list, device)
    rota_list = rota_list_init(teacher_list, device, quick_flag)

    # init eval() & train()
    for i in range(model_num):
        teacher_list[i].eval()
        rota_list[i].train()
    
    print('Training the Rotavap ...')
    for epoch in range(num_epoch):
        running_loss = 0
        n_samples = 0
        for batch_num, (inputs, target) in enumerate(dataloader):
            batchSize = inputs.size(0)
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                features = []
                for i in range(model_num):
                    features.append(teacher_list[i](inputs))
            
            outputs = rota_list[0](features[0]).unsqueeze(2)
            for i in range(1, model_num):
                temp = rota_list[i](features[i]).unsqueeze(2)
                outputs = torch.cat((outputs,temp), dim=2)
            
            # outputs = rota_list[0](teacher_list[0](inputs)).unsqueeze(2)
            # for i in range(1, model_num):
            #     temp = rota_list[i](teacher_list[i](inputs)).unsqueeze(2)
            #     outputs = torch.cat((outputs,temp), dim=2)
            
            loss = torch.mean(torch.std(outputs, dim=2))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            running_loss += loss.item()

        epoch_loss = running_loss / n_samples
        print('Epoch {}, Loss :{:.8f}'.format(epoch, epoch_loss))

    return teacher_list, rota_list



