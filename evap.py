import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from .rota import rota_train

def feature_dist(fea_0, fea_1, T = 2):
    fea_0 = F.log_softmax(fea_0/T, dim=1)
    fea_1 = F.log_softmax(fea_0/T, dim=1)
    return nn.KLDivLoss(reduction='mean')(fea_0, fea_1)
    

def student_train(args, dataloader, criterion, retcher_list, student, device):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag
    '''
    optimizer_stu = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-4)
    student_ft = nn.Sequential(*(list(student.children())[:-1]))
    student_fc = nn.Sequential(*(list(student.children())[-1:]))

    for this_model in retcher_list:
        this_model.eval()
    student.train()

    print('Distilling by Rotavap ...')
    for epoch in range(args.num_epoch):
        running_loss = 0
        running_loss_plain = 0
        n_samples = 0
        for batch_num, (inputs, labels) in enumerate(dataloader):
            batchSize = inputs.size(0)
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_stu.zero_grad()

            features_stu = student_ft(inputs)
            outputs_stu = student_fc(features_stu)
            loss = criterion(outputs_stu, labels)
            loss_plain = loss.item()
            
            with torch.no_grad():
                features = []
                for this_model in retcher_list:
                    features.append(this_model(inputs))
            
            for this_fea in features:
                loss += feature_dist(features_stu, this_fea)
            
            loss.backward()
            optimizer_stu.step()
            # scheduler.step()
            running_loss += loss.item()
            running_loss_plain += loss_plain

        epoch_loss = running_loss / n_samples
        epoch_loss_plain = running_loss_plain / n_samples

        print('Epoch {}, Loss : {:.8f}\nPlain Loss : {:.8f}'.format(epoch, epoch_loss, epoch_loss_plain))

    return student

def evap(args, dataloader, model_list, student, device):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag
    '''
    model_num = len(model_list)
    teacher_list, rota_list = rota_train(args, dataloader, model_list, device, args.quick_flag)
    retcher_list = []
    for i in range(model_num):
        retcher_list.append(nn.Sequential(teacher_list[i], rota_list[i]))

    criterion = nn.CrossEntropyLoss()
    student = student_train(args, dataloader, criterion, retcher_list, student, device)

