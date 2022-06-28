import math
import numpy as np

import torch
torch.set_grad_enabled(False)
from torch import empty

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def dataset_generator(nb_points):
   
    x = torch.rand(nb_points)
    y = torch.rand(nb_points)
    
# Labelization: Label '1' if the point is inside the circle, '0' if outside
    cercle_center=torch.tensor([[0.5],[0.5]])
    label_0=[]
    label_1=[]
    label_0_use=[]

    a = torch.square(x-cercle_center[0])
    b = torch.square(y-cercle_center[1])

    R = 1/math.sqrt(2*math.pi)

    for i in range(len(x)):

        if a[i] + b[i] <= R*R:
            label_1.append([x[i],y[i]])
            label_0_use.append(0)

        if a[i] + b[i] > R*R:
            label_0.append([x[i],y[i]])
            label_0_use.append(1)


    return  x,y,label_0_use



def dataset_plot(x,y,label,ax):
    colors = []
    
    for l in labels:
        if (l==1):
            colors.append('r')
        else:
            colors.append('b')
            
    ax.scatter(x[:,0],y[:,1],color = colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")   
    
    
    
#x,y,label = dataset_generator(1000)

#train_label = label.max(dim = 1)[1].long()
#fig_train, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8), sharex=True)
#plot_with_labels(x,y, train_label, axes)













    
    
    