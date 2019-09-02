'''
@Description: 
@Version: 1.0
@Author: Taoye Yin
@Date: 2019-08-17 15:59:28
@LastEditors: Taoye Yin
@LastEditTime: 2019-08-28 17:17:22
'''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1) #dim:指明维度，dim=0表示按列计算；dim=1表示按行计算

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_classes = self.fc2(x)
        x_classes = self.softmax1(x_classes)

        return x_classes
print (Net())