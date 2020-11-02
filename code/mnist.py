import argparse
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pdb
import subprocess as sp
import requests.api
import re
import multiprocessing
import sys
import datetime
import numpy as np

cuda = torch.device('cuda')
cpu = torch.device('cpu')
torch.cuda.synchronize(device=cuda)

class mnist(nn.Module):

    def __init__(self, layerGPU, numInputChannels, numOutputChannels, K):
        super(mnist, self).__init__()
        self.layerGPU = layerGPU
        self.numInputChannels = numInputChannels
        self.numOutputChannels = numOutputChannels
        self.K = K

        assert(layerGPU <= 3 and layerGPU >= 1)
        assert(isinstance(layerGPU, int))

        if (self.layerGPU==1):
            self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0).to(device=cuda)
            self.activation1 = nn.BatchNorm2d(32)
            self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
        else:
            self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0)
            self.activation1 = nn.BatchNorm2d(32)
            self.ch = 32 # ch --> what the number of input channels for the next layer will be

        self.out1 = nn.ReLU()
        
        if (self.layerGPU==2):
            self.layer2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(6,6),stride=2, padding=0).to(device=cuda)
            self.activation2 = nn.BatchNorm2d(32)
            self.ch=self.numOutputChannels # ch --> what the number of input channels for the next layer will be
        else:
            self.layer2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(6,6),stride=2, padding=0)
            self.activation2 = nn.BatchNorm2d(32)
            self.ch=32 # ch --> what the number of input channels for the next layer will be

        self.out2 = nn.ReLU()

        if (self.layerGPU==3):
            self.layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6,6), stride=2, padding=0).to(device=cuda)
        else:
            self.layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6,6), stride=2, padding=0)

        self.activation3 = nn.Sigmoid()

    def forward(self, x):

        if (self.layerGPU==1):
            out1_ = self.layer1(x.to(device=cuda))
            out1_ = self.activation1(out1_.to(device=cpu))
        else:
            out1_ = self.layer1(x)
            out1_ = self.activation1(out1_)

        out1 = self.out1(out1_)
        
        if (self.layerGPU==2):
            out2_ = self.layer2(out1.to(device=cuda))
            out2_ = self.activation2(out2_.to(device=cpu))
        else:
            out2_ = self.layer2(out1)
            out2_ = self.activation2(out2_)

        out2 = self.out2(out2_)

        if (self.layerGPU==3):
            out3_ = self.layer3(out2.to(device=cuda))
            out3 = self.activation3(out3_.to(device=cpu))
        else:
            out3_ = self.layer3(out2)
            out3 = self.activation3(out3_)
        
        return(out3.detach().numpy())


def setToZero(model, layer, percentage):
    # elementDict = model.state_dict()
    # for thing in elementDict:
    #     print(thing, elementDict[thing].shape)
    percentage = percentage*0.01
    if percentage <= 0:
        return None
    name = f"layer{layer}.weight"
    elementDict = model.state_dict()

    for param_tensor in elementDict:
        print(param_tensor, "\t", elementDict[param_tensor].size())
    WeightTensornumber = np.prod(np.array(elementDict[name].shape))
    newLayer = np.random.random_sample(WeightTensornumber)
    layerQuantile = np.quantile(newLayer, percentage)
    binaryAns = newLayer < layerQuantile
    newLayer[binaryAns] = 0

    sparseTensor = torch.Tensor(newLayer.reshape(np.array(elementDict[name].shape)))
    elementDict[name] = sparseTensor
    # elementDict[layer] = torch.from_numpy(sparseTensor).float()
    model.load_state_dict(elementDict)
    return model

def runMnist(batchSize, layer, sparcity):

    if layer == 1:
        net = mnist(1,10,32,3)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 10, 1, 1)
        print("execute")
        image = net(inp)
        
    if layer == 2:
        net = mnist(2,32,32,3)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 10,1,1)
        print("execute")
        image = net(inp)

    if layer == 3:
        net = mnist(3,32,1,5)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 10,1,1)
        # print(inp.shape)
        print("execute")
        image = net(inp)
    
    test = net.state_dict()
    print(test[f"layer{layer}.weight"])

def main():

    # Get input parameters
    parser = argparse.ArgumentParser(description='TorchDCNN for deployment on GPU')
    parser.add_argument('layer', help='layer of model')
    parser.add_argument('Percent_Sparcity', help="'mnist' or 'celeba'")
    parser.add_argument('iteration')

    # Convert to Python variables
    args = parser.parse_args()
    variables = vars(args)
    layer = int(variables["layer"])
    temp = float(variables["Percent_Sparcity"])
    iterrr = float(variables["iteration"])


    # print(type(temp), type(0.5))
    print(f"iteration:{iterrr} mnist layer:{layer}, sparcity:{temp}")

    runMnist(1, layer, temp)

if __name__ == '__main__':
    main()

