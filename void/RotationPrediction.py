from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import random
import numpy as np
import warnings
import torch.utils.data as data
from PIL import Image

plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

class RotationDataset(Dataset):

    def __init__(self, split, unlabelled_root_dir, labelled_root_dir, preTransform=None, postTransform=None):
        self.unlabelled_root_dir = unlabelled_root_dir
        self.labelled_root_dir = labelled_root_dir

        # Output of pretransform should be PIL images
        self.preTransform = preTransform

        self.postTransform = postTransform
        self.split = split
        self.labelled_data_dir = labelled_root_dir + '/' + split
        self.labelled_dataset = datasets.ImageFolder(self.labelled_data_dir, self.preTransform)
        if split == 'train':
            self.unlabelled_dataset = datasets.ImageFolder(self.unlabelled_root_dir, self.preTransform)
            self.dataset = ConcatDataset([self.labelled_dataset, self.unlabelled_dataset])
        else:
            self.dataset = self.labelled_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img0, clss = self.dataset[idx]

        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_img = img0.rotate(rot_angle)
        if self.postTransform:
            sample = self.postTransform(rot_img)
        else:
            sample = transforms.ToTensor(rot_img)
        return sample, rot_class

        # #Rotate PIL image multiple times
        # img1 = img0.rotate(90)
        # img2 = img0.rotate(180)
        # img3 = img0.rotate(270)

        # img_list = [img0,img1,img2,img3]

        # arr4 = np.arange(4)
        # np.random.shuffle(arr4)
        # img_newList = [img_list[arr4[0]], img_list[arr4[1]], img_list[arr4[2]], img_list[arr4[3]]]

        # if self.postTransform:
        #     sampleList = list(map(lambda pilim : self.postTransform(pilim),img_newList))
        # else:
        #     sampleList = list(map(lambda pilim : transforms.ToTensor(pilim),img_newList))
        # sample = torch.stack(sampleList)
        # rotation_labels = torch.LongTensor(arr4)
        # return sample, rotation_labels


# General Code for supervised train
def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, f, verbIter, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        f.write('-' * 10)
        f.write('\n')
        f.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            end = time.time()

            # Iterate over data.
            for batch_num, (inputs, labels) in enumerate(dataloaders[phase]):
                # #Reshaping the inputs and labels
                # shapeList = list(inputs.size())[1:]
                # shapeList[0] = -1
                # inputs = torch.reshape(inputs, shapeList)
                # labels = torch.reshape(labels, (-1,))

                data_time = time.time() - end
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = inputs.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train
                forward_start_time = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                forward_time = time.time() - forward_start_time

                # statistics
                running_loss += loss.item() * inputs.size(0)
                pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

                if batch_num+1 % verbIter == 0:
                    # Metrics
                    top_1_acc = running_corrects / n_samples
                    epoch_loss = running_loss / n_samples

                    f.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))
                    f.write('Full Batch time: {} , Data load time: {} , Forward time: {}\n'.format(time.time() - end,
                                                                                                   data_time,
                                                                                                   forward_time))
                    f.flush()

                end = time.time()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples

            f.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))
            f.flush()

            # deep copy the model
            if phase == 'test' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    f.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('Best test Acc: {:4f} \n'.format(best_acc))
    f.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--netCont', default='', help="path to net (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

out_dir = './models'
log_out_dir = './logs'

try:
    os.makedirs(out_dir)
    os.makedirs(log_out_dir)
except OSError:
    pass

f = open("{}/training_logs.txt".format(log_out_dir), "w+")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
f.write("Random Seed: {} \n".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
image_size = (224, 224)
labelled_data_root = '../task2/Skin40_data_split/split{}'.format(0)
unlabelled_data_root = '../datasets/unlabelled_skin'
batch_size = 60
epochs = 120
# Initiate dataset and dataset transform
data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
    ]),
}

data_post_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
}

image_datasets = {x: RotationDataset(x, unlabelled_data_root,labelled_data_root, data_pre_transforms[x], data_post_transforms[x]) for x in
                  ['train', 'test']}
assert image_datasets
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=True,
                                   num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f.write("using " + str(device) + "\n")
f.flush()

# Model Initialization
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

if opt.netCont != '':
    model_ft.load_state_dict(torch.load(opt.netCont, map_location=device))
    f.write('Loaded state and continuing training')

# Model trainer
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_ft.parameters(), lr=opt.lr, momentum=0.9, nesterov=True, weight_decay=0.005)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=[80, 100, 110], gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_conv, exp_lr_scheduler, device, out_dir, f, 1000,
                       num_epochs=epochs)
torch.save(model_ft.state_dict(), './models/skin_rotation_pretrained.pth')
f.close()