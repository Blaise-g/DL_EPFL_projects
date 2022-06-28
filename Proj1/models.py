# -*- coding: utf-8 -*-
""" Define the three architectures.
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class Base_net(nn.Module):
    """
    Baseline ConvNet similar to LeNet-5 architecture.
    """
    def __init__(self):
        super(Base_net, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(90)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 90)
        self.fc3 = nn.Linear(90, 2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),  kernel_size=2, stride=2)
        
        x = F.relu(self.bn3(self.fc1(x.view(x.size(0), -1))))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

class Siamese_net_ws(nn.Module):
    """
    Siamese ConvNet with weight sharing implementation, each of the two branch has the same architecture as the baseline
    ConvNet.
    """
    def __init__(self):
        super(Siamese_net_ws, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(90)
        self.bn5=nn.BatchNorm1d(10)
        self.bn6=nn.BatchNorm1d(90)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 90)
        self.fc3 = nn.Linear(90,10)
        self.fc4 = nn.Linear(20, 90)
        self.fc5 = nn.Linear(90, 2)
        
    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x[:, 0].view(-1, 1, 14, 14)))), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.bn1(self.conv1(x[:, 1].view(-1, 1, 14, 14)))), kernel_size=2, stride=2)

        x1 = F.max_pool2d(F.relu(self.bn2(self.conv2(x1))),  kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.bn2(self.conv2(x2))),  kernel_size=2, stride=2)
        
        x1 = F.relu(self.bn3(self.fc1(x1.view(x1.size(0), -1))))
        x2 = F.relu(self.bn3(self.fc1(x2.view(x2.size(0), -1))))
        
        x1 = F.relu(self.bn4(self.fc2(x1)))
        x2 = F.relu(self.bn4(self.fc2(x2)))
        
        x1 = F.relu(self.bn5(self.fc3(x1)))
        x2 = F.relu(self.bn5(self.fc3(x2)))
        
        x = F.relu(self.bn6(self.fc4(torch.cat((x1, x2), dim=1))))
        x = self.fc5(x)
        return x

class Siamese_net_ws_aux(nn.Module):
    """
    Siamese ConvNet with weight sharing implementation and auxiliary losses.
    """
    def __init__(self):
        super(Siamese_net_ws_aux, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(90)
        self.bn5=nn.BatchNorm1d(10)
        self.bn6=nn.BatchNorm1d(90)
        
        #self.dropout = nn.Dropout(0.25) dropout was also exp
        
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 90)
        self.fc3 = nn.Linear(90,10)
        self.fc4 = nn.Linear(20, 90)
        self.fc5 = nn.Linear(90, 2)
        
    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x[:, 0].view(-1, 1, 14, 14)))), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.bn1(self.conv1(x[:, 1].view(-1, 1, 14, 14)))), kernel_size=2, stride=2)

        x1 = F.max_pool2d(F.relu(self.bn2(self.conv2(x1))),  kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.bn2(self.conv2(x2))),  kernel_size=2, stride=2)
        
        
        x1 = F.relu(self.bn3(self.fc1(x1.view(x1.size(0), -1))))
        x2 = F.relu(self.bn3(self.fc1(x2.view(x2.size(0), -1))))
        
        x1 = F.relu(self.bn4(self.fc2(x1)))
        x2 = F.relu(self.bn4(self.fc2(x2)))
        
        x1_aux = self.fc3(x1)
        x2_aux = self.fc3(x2)
        
        x1 = F.relu(self.bn5(x1_aux))
        x2 = F.relu(self.bn5(x2_aux))
        x = F.relu(self.bn6(self.fc4(torch.cat((x1, x2), dim=1))))
        x = self.fc5(x)
        
        return x, x1_aux, x2_aux