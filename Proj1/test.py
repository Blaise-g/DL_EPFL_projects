# -*- coding: utf-8 -*-
""" Main test file to run without calling arguments
"""

import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue
from models import *
from helpers import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt #just for plotting
import numpy as np #just for plotting
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    #controlling sources of randomness
    torch.manual_seed(0)
    
    #setting number of total number of samples, mini batch size and rounds
    N = 1000
    rounds = 15

    #generate the train and test datasets
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)

    #check dimensions
    print('The shape of inputA and B is : {}'.format(train_input.shape))
    print('The shape of targetA and B is : {}'.format(train_target.shape))
    print('The shape of classes A and B is : {}'.format(train_classes.shape))

    #scale data
    train_input, test_input = normalize_data(train_input, test_input)

    #run the models with the best hyper-parameters found
    #for the baseline nb_epochs=50, mini_batch_size=50, lr=0.1
    print("--------------------------------------------------")
    print("ConvNet model with no weight sharing nor auxiliary losses")
    model_nows_noaux, loss_per_round_nows_noaux, test_errors_nows_noaux = BasePipeline(Base_net(), 50, rounds, N,  0.1, 50, train_input, train_target, train_classes, test_input, test_target, test_classes)
    perf_summary(test_errors_nows_noaux)
    print("The total number of trainable parameters of this Model:", sum(p.numel() for p in model_nows_noaux.parameters() if p.requires_grad),'\n')
    
    #for the siamese with weight sharing implementation nb_epochs=50, mini_batch_size=100, lr=0.01
    print("--------------------------------------------------")
    print("Siamese ConvNet model with weight sharing and no auxiliary losses")
    model_ws_noaux, loss_per_round_ws_noaux, test_errors_ws_noaux = Ws_Pipeline(Siamese_net_ws(), 100, rounds, N, 0.01, 50, train_input, train_target, train_classes, test_input, test_target, test_classes)
    perf_summary(test_errors_ws_noaux)
    print("The total number of trainable parameters of this Model:", sum(p.numel() for p in model_ws_noaux.parameters() if p.requires_grad),'\n')
    
    #for the siamese with weight sharing implementation nb_epochs=50, mini_batch_size=100, lr=0.01
    print("--------------------------------------------------")
    print("Siamese ConvNet model with weight sharing and auxiliary losses")
    model_ws_aux, loss_per_round_ws_aux, test_errors_ws_aux = Ws_aux_Pipeline(Siamese_net_ws_aux(), 100, rounds, N, 0.01, 50, train_input, train_target, train_classes, test_input, test_target, test_classes)
    perf_summary(test_errors_ws_aux)
    print("The total number of trainable parameters of this Model:", sum(p.numel() for p in model_ws_aux.parameters() if p.requires_grad),'\n')

    #plots
    std_1,best_acc_1 = avg_acc_std(test_errors_nows_noaux)
    std_2,best_acc_2 = avg_acc_std(test_errors_ws_noaux)
    std_3,best_acc_3 = avg_acc_std(test_errors_ws_aux)

    stds = [std_1, std_2, std_3]
    best_accs = [best_acc_1,best_acc_2,best_acc_3]

    plot_err_evolution(test_errors_nows_noaux,test_errors_ws_noaux,test_errors_ws_aux)
    plot_acc_std(best_accs, stds)