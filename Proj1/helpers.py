# -*- coding: utf-8 -*-
""" Helpers functions for the main test run.
"""

import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt #just for plotting
import numpy as np #just for plotting

def normalize_data(train_data_input, test_data_input):
    """
    Scale data input to have zero mean and unit variance based on the train_data_input tensor.
    """
    mu, std = train_data_input.mean(), train_data_input.std()
    train_data_input.sub(mu).div(std)
    test_data_input.sub(mu).div(std)
    return train_data_input, test_data_input

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs, learning_rate, verbose=False):
    """
    Train ConvNet model without auxiliry losses.
    """
    model,criterion = model,nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #move model and criterion to gpu if CUDA available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    losses = []
    for e in range(nb_epochs):
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            # forward pass: compute prediction
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            #loss
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                
            # backward pass
            model.zero_grad()
            loss.backward()
            
            #update weights
            optimizer.step()
            
            losses.append(loss.data.item())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()         
            
        if verbose:
            print("Epoch: {} \t -> Loss: {} ".format(e, losses))
    return losses

def train_model_auxiliaryloss(model, train_input, train_class, train_target, mini_batch_size, nb_epochs, learning_rate, verbose=False):
    """
    Train ConvNet model with auxiliry losses.
    """
    model,criterion = model,nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #move model and criterion to gpu if CUDA available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    losses_aux = []
    for e in range(nb_epochs):
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            # Forward pass: compute prediction
            output_primary, output_aux1, output_aux2 = model(train_input.narrow(0, b, mini_batch_size))
            
            # Main loss + 2 auxiliary losses
            loss =  criterion(output_primary, train_target.narrow(0, b, mini_batch_size)) + criterion(output_aux1, train_class[:, 0].narrow(0, b, mini_batch_size)) + criterion(output_aux2, train_class[:, 1].narrow(0, b, mini_batch_size))
            
            # backward pass
            model.zero_grad()
            loss.backward()
            
            #update weights
            optimizer.step()
            
            losses_aux.append(loss.data.item())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()         
            
        if verbose:
            print("Epoch: {} \t -> Loss: {} ".format(e, losses))
    return losses_aux

def compute_errors_aux(model, data_input, data_target, mini_batch_size):
    """
    Compute number of target errors of model with auxiliary losses.
    """
    nb_target_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        # Prediction
        output_primary, output_aux1, output_aux2 = model(data_input.narrow(0, b, mini_batch_size))
        highest_numbers_indices_main = output_primary.max(1)[1]
        highest_numbers_indices_aux1 = output_aux1.max(1)[1]
        highest_numbers_indices_aux2 = output_aux2.max(1)[1]
        for i in range(highest_numbers_indices_main.size(0)):
            if highest_numbers_indices_main[i] != data_target[b + i]:
                nb_target_errors += 1
    return nb_target_errors

def compute_errors(model, data_input, data_target, mini_batch_size):
    """
    Compute number of target errors of model with no auxiliary losses.
    """
    nb_target_errors = 0
    
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        highest_numbers_indices_main = output.max(1)[1]
        for i in range(highest_numbers_indices_main.size(0)):
            if highest_numbers_indices_main[i] != data_target[b + i]:
                nb_target_errors += 1
    return nb_target_errors

def BasePipeline(model, mini_batch_size, rounds, N, learning_rate, nb_epochs, train_input, train_target, train_classes, test_input, test_target, test_classes):
    """
    Full pipeline with baseline model.
    """

    loss_per_round = []
    test_errors_list = []
    
    for k in range(rounds):
        
        print('Starting Round', k+1)
        
        #if cuda available move to gpu
        if torch.cuda.is_available():
            train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
            test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()
        
        # Model training
        losses = train_model(model, train_input, train_target, mini_batch_size, nb_epochs, learning_rate)
        loss_per_round.append(losses)

        # Predict and compute error
        nb_test_errors = compute_errors(model, test_input, test_target, mini_batch_size)
        test_errors_list.append(nb_test_errors/test_input.size(0))

        print('Target error rate: {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
        print("--------------------------------------------------\n")
    return model, loss_per_round, test_errors_list

def Ws_Pipeline(model, mini_batch_size, rounds, N, learning_rate, nb_epochs, train_input, train_target, train_classes, test_input, test_target, test_classes):
    """
    Full pipeline with weight sharing and no auxiliary losses.
    """
    loss_per_round = []
    test_errors_list = []
    
    for k in range(rounds):
        
        print('Starting Round', k+1)
        
        if torch.cuda.is_available():
            train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
            test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()
        
        # Model training
        losses = train_model(model, train_input, train_target, mini_batch_size, nb_epochs, learning_rate)
        loss_per_round.append(losses)

        # Predict and compute error
        nb_test_errors = compute_errors(model, test_input, test_target, mini_batch_size)
        test_errors_list.append(nb_test_errors/test_input.size(0))

        # Logging
        print('Target error rate: {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
        print("--------------------------------------------------\n")
    return model, loss_per_round, test_errors_list

def Ws_aux_Pipeline(model, mini_batch_size, rounds, N, learning_rate, nb_epochs, train_input, train_target, train_classes, test_input, test_target, test_classes):
    """
    Full pipeline with weight sharing and auxiliary losses.
    """
    
    loss_per_round = []
    test_errors_list = []
    
    for k in range(rounds):
        
        print('Starting Round', k+1)
        
        if torch.cuda.is_available():
            train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
            test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()
        
        # Model training
        losses = train_model_auxiliaryloss(model, train_input, train_classes, train_target, mini_batch_size, nb_epochs, learning_rate)
        loss_per_round.append(losses)

        # Predict and compute error
        nb_test_errors = compute_errors_aux(model, test_input, test_target, mini_batch_size)
        test_errors_list.append(nb_test_errors/test_input.size(0))

        # Logging
        print('Target error rate: {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
        print("--------------------------------------------------\n")
    return model, loss_per_round, test_errors_list

def perf_summary(test_errors_list, rounds=15):
    """
    Get mean and standard devation of the test errors during all the rounds for one model.
    """
    testErrorMean = torch.FloatTensor(test_errors_list).mean().item()
    testErrorStd = torch.FloatTensor(test_errors_list).std().item()
    print("Estimates of {} rounds:".format(rounds))
    print("Test error Average: {:3f};  Test error standard deviations: {:3f}".format(testErrorMean, testErrorStd))

def avg_acc_std(test_errors_list):
    """
    Get standard devation of the test errors during all the rounds and the best test error rate for one model.
    """
    std = torch.FloatTensor(test_errors_list).std().item()
    avg_err = torch.FloatTensor(test_errors_list).min().item()
    return std, avg_err

def plot_err_evolution(test_errors_1, test_errors_2, test_errors_3):
    """
    Plot the evolution of the test error during the different rounds for each model
    """
    plt.figure(figsize=(10, 5))
    ax2 = plt.subplot(111)
    ax2.plot(range(1, len(test_errors_1)+1), test_errors_1, label='Baseline', marker='^')
    ax2.plot(range(1, len(test_errors_2)+1), test_errors_2, label='Siamese', marker='^')
    ax2.plot(range(1, len(test_errors_3)+1), test_errors_3, label='Siamese with aux. losses', marker='^')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Round')
    ax2.set_title('Test error rate in function of rounds')
    ax2.legend()
    plt.savefig('Test__error_rate_over_rounds.png')
    
def plot_acc_std(avg_accs, stds):
    """
    Bar plot of the average test error rate with the corresponding standard deviation over all rounds for each model.
    """
    
    models = ['Baseline', 'Siamese', 'Siamese wit aux. losses']
    x_pos = np.arange(len(models))  #illegal use of numpy
    
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, avg_accs, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=5, width=0.3, color=['pink', 'blue', 'cyan'])
    ax.set_ylabel('Average test error rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_title('Average test error rate and standard deviation for each model')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_avg_test_error_rate_std.png')
    plt.show()
