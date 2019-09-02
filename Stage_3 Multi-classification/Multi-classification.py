'''
@Description: 
@Version: 1.0
@Author: Taoye Yin
@Date: 2019-08-17 15:59:28
@LastEditors: Taoye Yin
@LastEditTime: 2019-08-29 14:34:40
'''
# It's empty. Surprise!
# Please complete this by yourself.
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from Multi_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy
import sys
os.chdir(sys.path[0])
ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path): #一个完整的路径
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])
        label_classes = int(self.file_info.iloc[idx]['classes'])

        sample = {'image': image, 'species': label_species,'classes': label_classes}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(),
                                       ])  #?0.5??????????PIL??
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
visualize_dataset()

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list_species = {'train': [], 'val': []}
    Loss_list_classes = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss_classes,running_loss_species = 0.0 , 0.0
            corrects_classes,corrects_species = 0,0

            for idx,data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_species = data['species'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad() #每一个batch会累加梯度，要清零

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)
                    x_classes = x_classes.view(-1,2)
                    x_species = x_species.view(-1,3)

                    _, preds_classes = torch.max(x_classes, dim = 1) #列作最大值，出来的是行向量
                    __, preds_species = torch.max(x_species, dim = 1)
                    loss_classes = criterion(x_classes, labels_classes)
                    loss_species = criterion(x_species, labels_species)
                    
                    if phase == 'train':
                        loss_classes.backward(retain_graph=True)
                        optimizer.step()
                    if phase == 'train':
                        loss_species.backward()
                        optimizer.step()
                # print (inputs.size(0))
                running_loss_classes += loss_classes.item() * inputs.size(0) # 为什么要乘以size(0) batch训练得问题 inputs.size(0) = 1
                running_loss_species += loss_species.item() * inputs.size(0)
                
                corrects_classes += torch.sum(preds_classes == labels_classes)  # 统计true得值
                corrects_species += torch.sum(preds_species == labels_species)
                
            # exp_lr_scheduler.step()  #在epoch里面
            epoch_loss_classes = running_loss_classes / len(data_loaders[phase].dataset)
            Loss_list_classes[phase].append(epoch_loss_classes)
            epoch_loss_species = running_loss_species / len(data_loaders[phase].dataset)
            Loss_list_species[phase].append(epoch_loss_species)
            
            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)
            
            epoch_acc = epoch_acc_species + epoch_acc_classes
            
            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            
            print('{} Loss_classes: {:.4f}  Acc_species_classes: {:.2%}'.format(phase, epoch_loss_classes,epoch_acc_classes))
            print('{} Loss_species: {:.4f}  Acc_species_species: {:.2%}'.format(phase, epoch_loss_species,epoch_acc_species))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species + classes Acc: {:.2%}'.format(best_acc)) #怎么降这两个loss同时尽量取最优
            

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val species + classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list_classes , Loss_list_species ,Accuracy_list_classes, Accuracy_list_species

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs learningrate startgy
model, Loss_list_classes , Loss_list_species ,Accuracy_list_classes, Accuracy_list_species = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)


x = range(0, 100)
y1 = Loss_list_classes["val"]
y2 = Loss_list_classes["train"]

plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss_classes')
plt.savefig("train and val loss vs epoches_classes.jpg")
plt.close('all') 


y3 = Accuracy_list_classes["train"]
y4 = Accuracy_list_classes["val"]

plt.plot(x, y3, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y4, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Classes acc vs. epoches')
plt.ylabel('Classes accuracy')
plt.savefig("train and val Classes acc vs epoches.jpg")
plt.close('all')

y5 = Loss_list_species["val"]
y6 = Loss_list_species["train"]

plt.plot(x, y5, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y6, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches_species')
plt.ylabel('loss_species')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all') 

y7 = Accuracy_list_species["train"]
y8 = Accuracy_list_species["val"]

plt.plot(x, y7, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y8, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Species acc vs. epoches')
plt.ylabel('Species accuracy')
plt.savefig("train and val Species acc vs epoches.jpg")
plt.close('all')
# torch.save(model.state_dict(), 'E:/params.pkl')
######################################## Visualization ##################################
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)
            labels_species = data['species'].to(device)

            x_classes, x_species = model(inputs.to(device))
            x_classes = x_classes.view(-1,2)
            x_species = x_species.view(-1,3) #numpy = resize
            _, preds_species = torch.max(x_species, 1)#(tensor, dim) return tensor, indices
            __, preds_classes = torch.max(x_classes, 1)
            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted species: {} ground-truth species:{}\n predicted classes: {} ground-truth classes:{}'.format(SPECIES[preds_species],SPECIES[labels_species],CLASSES[preds_classes],CLASSES[labels_classes]))
            plt.show()

visualize_model(model)