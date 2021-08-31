import torch
import torch.nn as nn
import torch.optim as optim

def feature_dist(fea_0, fea_1):
    temp = nn.KLDivLoss(reduction='none')(fea_0, fea_1)
    return torch.mean(temp, dim=1)

def count_loss_weight(loss, weight):
    model_num = len(loss)
    Ret = []
    for i in range(model_num):
        Ret.append(loss[i] * weight[i])

    sumsum = Ret[0]
    for i in range(1, model_num):
        sumsum += Ret[i]

    for i in range(model_num):
        Ret[i] /= sumsum
    
    return Ret

def student_train(args, dataloader, criterion, teacher_list, student, device):
    '''
    including args
    args : lr, momentum, num_epoch, quick_flag, alpha
    '''
    model_num = len(teacher_list)
    optimizer_stu = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-4)

    for this_model in teacher_list:
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

            outputs_stu = student(inputs)
            loss_0 = criterion(outputs_stu, labels)
            loss_plain = loss_0.item()
            
            with torch.no_grad():
                outputs_tea = []
                weight = []
                for i, this_model in enumerate(teacher_list):
                    this_outputs = this_model(inputs)
                    outputs_tea.append(this_outputs)
                    weight.append(criterion(this_outputs, labels))
                    
                    pred_top_1 = torch.topk(this_outputs, k=1, dim=1)[1]
                    weight[i] *= pred_top_1.eq(labels.view_as(pred_top_1)).int()

            loss_1 = []
            for this_fea in outputs_tea:
                loss_1.append(feature_dist(outputs_stu, this_fea))
            
            loss_main = loss_0 * (1-args.alpha)
            loss_weight = count_loss_weight(loss_1, weight)
            for i in range(model_num):
                loss_main += torch.mean(loss_weight[i]) * args.alpha

            loss_main.backward()
            optimizer_stu.step()
            # scheduler.step()
            running_loss += loss_main.item()
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

    criterion = nn.CrossEntropyLoss()
    student = student_train(args, dataloader, criterion, model_list, student, device)

