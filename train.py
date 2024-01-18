from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, datasets
import matplotlib.pyplot as plt
import time, os, copy, random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

class CustomHead(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(CustomHead, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, in_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_features),
            nn.SiLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# seed fix
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='exp')
parser.add_argument('--data_dir', default='R')
parser.add_argument('--pre', action='store_true')
parser.add_argument('--drop_out', type=float, default=0.0)
opt = parser.parse_args()
opt.name = f'{opt.name}_{opt.drop_out}'
## create folder runs
os.makedirs('runs', exist_ok=True)
writer = SummaryWriter(f'runs/{opt.name}')
print('Using pre-trained Model? : ', opt.pre)

# os.environ["CUDA_VISIBLE_DEVICE"] = '0,1,2,3'
device = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = torch.cuda.is_available()
model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=opt.pre, nclass=3).to(device) # pre-trained load
head = CustomHead(1280, 3, opt.drop_out).to(device)
model.head = head

# model = nn.DataParallel(model, output_device=0)
data_dir = f'{opt.data_dir}'
dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose([transforms.ToTensor()]))
dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose([transforms.ToTensor()]))

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=2) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        time1 = time.time()
        # Early Stopping code apply
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if (phase == 'train'):
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            if (phase == 'val'):
                writer.add_scalars(f'Loss/{opt.name}', {'train':train_loss[epoch],'val':epoch_loss}, epoch)
                writer.add_scalars(f'Accuracy/{opt.name}', {'train':train_acc[epoch],'val':epoch_acc}, epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print(f'learning rate : {scheduler.optimizer.param_groups[0]["lr"]}')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            time2= time.time()
            print('time:', time2 - time1)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Test Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criterion = nn.CrossEntropyLoss().cuda()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.97)
model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
if not os.path.exists('model'):
    os.mkdir('model')
torch.save(model_ft.state_dict(), f'model/{opt.name}.pt')