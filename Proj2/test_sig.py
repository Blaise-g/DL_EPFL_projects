import math
import torch
torch.set_grad_enabled(False)
from torch import empty
import numpy as np

import sys
sys.path.insert(0,'src')

from utils import dataset_generator

nb_training_points=1000
nb_testing_points=1000

x_train,y_train,label_train = dataset_generator(nb_training_points)
x_test,y_test,label_test = dataset_generator(nb_testing_points)


class Connection:
    
    def __init__(self, Neuron_connection):
        
        self.Neuron_connection = Neuron_connection
        self.weight = torch.rand(1)
        self.dWeight = torch.tensor([0])
        
        
        
class Neuron:
   
    eta = torch.tensor([0.01])
    alpha = torch.tensor([0.01])

    def __init__(self, layer):
        self.connectors = []
        self.error = torch.tensor([0])
        self.gradient = torch.tensor([0])
        self.output = torch.tensor([0])
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.connectors.append(con)

    def addError(self, err):
        self.error = self.error + err

    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return (math.exp(-x * 1.0)) / (1 + math.exp(-x * 1.0))**2
        ##return x*(1-x)


    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output
                
    def backward(self):
        self.gradient = self.error * self.dSigmoid(self.output);
        for connector in self.connectors:
            connector.dWeight = Neuron.eta * (connector.Neuron_connection.output * self.gradient) + self.alpha * connector.dWeight;
            connector.weight = connector.weight + connector.dWeight;
            connector.Neuron_connection.addError(connector.weight * self.gradient);
        self.error = 0;
        
    def forward(self):
        Output_sum = torch.tensor([0])
        if len(self.connectors) == 0:    
            return
        for connector in self.connectors:
            Output_sum = Output_sum +  connector.Neuron_connection.getOutput() * connector.weight; 
        self.output = self.sigmoid(Output_sum)


class Network:
    
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def forward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.forward();

    def backward(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backward()

    def LossMSE(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        #err = math.sqrt(err)
        return err


def main():
    
    topology = []
    
    topology.append(2)     # Input layer
    topology.append(25)    # Hidden layer 1
    topology.append(25)    # Hidden layer 2
    topology.append(25)    # Hidden layer 3
    topology.append(1)     # Output layer

    network = Network(topology)
    
    Neuron.eta = torch.tensor([0.01])             # Learning rate of the neurons 
    Neuron.alpha = torch.tensor([0.15])          # momentum factor of the neurons 
    
    
    print("")
    print(" ------------------- Training ------------------- ")    
    print("")
    print("- Learning rate: ", round(Neuron.eta.item(),3))
    print("- Activation function: Sigmoid ")
    print("- Loss function: MSE ")
    print("")
    
    outputs=torch.zeros([nb_training_points, 1])
    inputs=torch.zeros([nb_training_points,2])
    
    for i in range(nb_training_points):
        
        inputs[i][0]=x_train[i]
        inputs[i][1]=y_train[i]
        
        if label_train[i]==1:
            outputs[i]=0
            
        if label_train[i]==0:
            outputs[i]=1
      
    
    idx=0
    
    while (idx<7):           

        train_err = 0 
        idx+=1
    
        for i in range(len(inputs)):
            
            network.setInput(inputs[i])
            network.forward()
            network.backward(outputs[i])
            train_err = train_err + network.LossMSE(outputs[i])
        
        train_err_percent=round((train_err*100/nb_training_points).item(),1)
        print ("Training error: ", train_err_percent,"%","(",int(train_err.item()),"/",nb_training_points,")")                    
    
   
    
    

    print("")
    print(" ------------------- Testing ------------------- ")
    print("")
    
    outputs=torch.zeros([nb_testing_points, 1])
    inputs=torch.zeros([nb_testing_points,2])

    for i in range(nb_testing_points):

        inputs[i][0]=x_test[i]
        inputs[i][1]=y_test[i]

        if label_test[i]==1:
            outputs[i]=0

        if label_test[i]==0:
            outputs[i]=1
   
    test_err = 0 
    for i in range(len(x_test)):

        test_err = test_err + network.LossMSE(outputs[i]) 
        
    test_err_percent=round((test_err*100/nb_testing_points).item(),1)
    print ("Testing error: ", test_err_percent,"%","(",int(test_err.item()),"/",nb_testing_points,")")        

    