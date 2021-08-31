import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from rota import rota_train

def feature_dist(fea_0, fea_1, T = 2):
    # return nn.KLDivLoss(reduction='mean')(fea_0, fea_1)
    return nn.MSELoss()(fea_0, fea_1)
    

def student_train(args, dataloader, criterion, retcher_list, student, device):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag, classnum
    '''
    model_num = len(retcher_list)
    student_ft = nn.Sequential(*(list(student.children())[:-1]))
    optimizer_stu = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-4)
    
    for this_model in retcher_list:
        this_model.eval()
    student.train()

    print('Distilling by Rotavap ...')
    for epoch in range(args.num_epoch_1):
        running_loss = 0
        running_loss_plain = 0
        running_acc = 0
        n_samples = 0
        
        for batch_num, (inputs, labels) in enumerate(dataloader):
            batchSize = inputs.size(0)
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_stu.zero_grad()

            features_stu = nn.Flatten()(student_ft(inputs))
            outputs_stu = torch.softmax(student.fc(features_stu), dim=1)
            loss = criterion(outputs_stu, labels)
            loss_plain = loss.item()

            pred_top_1 = torch.topk(outputs_stu, k=1, dim=1)[1]
            this_acc = pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()
            
            with torch.no_grad():
                features = []
                for this_model in retcher_list:
                    features.append(nn.Flatten()(this_model(inputs)))
            
            for this_fea in features:
                loss += feature_dist(features_stu, this_fea) * args.alpha / model_num
            
            loss.backward()
            optimizer_stu.step()
            # scheduler.step()
            running_loss += loss.item() * batchSize
            running_loss_plain += loss_plain * batchSize
            running_acc += this_acc

        epoch_loss = running_loss / n_samples
        epoch_loss_plain = running_loss_plain / n_samples
        epoch_acc = running_acc / n_samples

        print('Epoch {}\nLoss : {:.8f}, Plain Loss : {:.8f}'.format(epoch, epoch_loss, epoch_loss_plain))
        print('Acc : {:.8f}'.format(epoch_acc))

    return student

def evap(args, dataloader, criterion, model_list, student, device):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag
    '''
    model_num = len(model_list)
    teacher_list, rota_list = rota_train(args, dataloader, model_list, device, args.quick_flag)
    retcher_list = []
    for i in range(model_num):
        retcher_list.append(nn.Sequential(teacher_list[i], nn.Flatten(), rota_list[i]))

    student = student_train(args, dataloader, criterion, retcher_list, student, device)
    return student

