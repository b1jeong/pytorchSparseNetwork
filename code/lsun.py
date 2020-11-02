import torch
import torch.nn as nn
import torch.nn.functional as F
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

class pgan(nn.Module):

    def __init__(self, layerGPU, varyLayerDim, numInputChannels, numOutputChannels, K):
        super(pgan, self).__init__()
        self.layerGPU = layerGPU
        self.numInputChannels = numInputChannels
        self.numOutputChannels = numOutputChannels
        self.K = K
        self.varyLayerDim = varyLayerDim

        assert(layerGPU <= 6 and layerGPU >= 1)
        assert(isinstance(layerGPU, int))

        # Layer 1 ConvTranspose
        if (self.layerGPU==1 and self.varyLayerDim==True):
            self.layer1 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=1).to(device=cuda) 
            self.activation1 = nn.BatchNorm2d(self.numOutputChannels)
        elif (self.layerGPU==1 and self.varyLayerDim==False):
            self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1).to(device=cuda)
            self.activation1 = nn.BatchNorm2d(512)
        else:
            self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
            self.activation1 = nn.BatchNorm2d(512)
        
        # Layer 1 RELU
        self.out1 = nn.ReLU()
        
        # Layer 2 ConvTranspose
        if (self.layerGPU==2 and self.varyLayerDim==True):
            self.layer2 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
            self.activation2 = nn.BatchNorm2d(self.numOutputChannels)
        elif (self.layerGPU==2 and self.varyLayerDim==False):
            self.layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
            self.activation2 = nn.BatchNorm2d(512)
        else:
            self.layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1)
            self.activation2 = nn.BatchNorm2d(512)
        
        # Layer 2 RELU
        self.out2 = nn.ReLU()

        # Layer 3 ConvTranspose
        if (self.layerGPU==3 and self.varyLayerDim==True):
            self.layer3 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
            self.activation3 = nn.BatchNorm2d(self.numOutputChannels)
        elif (self.layerGPU==3 and self.varyLayerDim==False):
            self.layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
            self.activation3 = nn.BatchNorm2d(512)
        else:
            self.layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1)
            self.activation3 = nn.BatchNorm2d(512)

        # Layer 3 RELU
        self.out3 = nn.ReLU()

        # Layer 4 ConvTranspose
        if (self.layerGPU==4 and self.varyLayerDim==True):
            self.layer4 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
            self.activation4 = nn.BatchNorm2d(self.numOutputChannels)
        elif (self.layerGPU==4 and self.varyLayerDim==False):
            self.layer4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
            self.activation4 = nn.BatchNorm2d(256)
        else:
            self.layer4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4),stride=2, padding=1)
            self.activation4 = nn.BatchNorm2d(256)
        # Layer 4 NORM

        # Layer 4 RELU
        self.out4 = nn.ReLU()

        # Layer 5 ConvTranspose
        if (self.layerGPU==5 and self.varyLayerDim==True):
            self.layer5 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
            self.activation5 = nn.BatchNorm2d(self.numOutputChannels)
        elif (self.layerGPU==5 and self.varyLayerDim==False):
            self.layer5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
            self.activation5 = nn.BatchNorm2d(128)
        else:
            self.layer5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4),stride=2, padding=1)
            self.activation5 = nn.BatchNorm2d(128)
        # Layer 5 NORM

        # Layer 5 RELU
        self.out5 = nn.ReLU()

        # Layer 6 ConvTranspose
        if (self.layerGPU==6 and self.varyLayerDim==True):
            self.layer6 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=1).to(device=cuda)
        elif (self.layerGPU==6 and self.varyLayerDim==False):
            self.layer6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1).to(device=cuda)
        else:
            self.layer6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1)

        # Layer 6 SIGMOID
        self.activation6 = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight)


    def forward(self, x):

        # Layer 1
        if (self.layerGPU==1):
            out1_ = self.layer1(x.to(device=cuda))
            out1_ = self.activation1(out1_.to(device=cpu))
        else:
            out1_ = self.layer1(x)
            out1_ = self.activation1(out1_)

        out1 = self.out1(out1_)
        
        # Layer 2
        if (self.layerGPU==2):
            out2_ = self.layer2(out1.to(device=cuda))
            out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
        else:
            out2_ = self.layer2(out1)
            out2_ = self.activation2(out2_)

        out2 = self.out2(out2_)

        # Layer 3
        if (self.layerGPU==3):
            out3_ = self.layer3(out2.to(device=cuda))
            out3_ = self.activation3(out3_.to(device=cpu))#.to(device=cpu)
        else:
            out3_ = self.layer3(out2)
            out3_ = self.activation3(out3_)

        out3 = self.out3(out3_)

        # Layer 4
        if (self.layerGPU==4):
            out4_ = self.layer4(out3.to(device=cuda))
            out4_ = self.activation4(out4_.to(device=cpu))#.to(device=cpu)
        else:
            out4_ = self.layer4(out3)
            out4_ = self.activation4(out4_)

        out4 = self.out4(out4_)

        # Layer 5
        if (self.layerGPU==5):
            out5_ = self.layer5(out4.to(device=cuda))
            out5_ = self.activation5(out5_.to(device=cpu))#.to(device=cpu)
        else:
            out5_ = self.layer5(out4)
            out5_ = self.activation5(out5_)

        out5 = self.out5(out5_)

        # Layer 6
        if (self.layerGPU==6):
            out6_ = self.layer6(out5.to(device=cuda))
            out6 = self.activation6(out6_.to(device=cpu))
        else:
            out6_ = self.layer6(out5)
            out6 = self.activation6(out6_)
        
        return(out6.detach().numpy())

def setToZero(model, layer, percentage):
    # elementDict = model.state_dict()
    # for thing in elementDict:
    #     print(thing, elementDict[thing].shape)
    percentage = percentage*0.01
    if percentage <= 0:
        return None
    name = f"layer{layer}.weight"
    elementDict = model.state_dict()


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


# def setToZero(model, layer, percentage):
#     print(layer)
#     if layer ==1:
#         elementDict = model.state_dict()
#         newLayer = np.random.random_sample(4194304)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(512, 512, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict)
#         return model.state_dict()[layer]
#     elif layer ==2:
#         elementDict = model.state_dict()
#         newLayer = np.random.random_sample(4194304)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(512, 512, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict)
#         return model.state_dict()[layer]
#     elif layer ==3:
#         elementDict = model.state_dict()
#         print(elementDict[layer])
#         newLayer = np.random.random_sample(4194304)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(512, 512, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict)
#         return model.state_dict()[layer]
#     elif layer ==4:
#         elementDict = model.state_dict()
#         newLayer = np.random.random_sample(2097152)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(512, 256, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict) 
#         return model.state_dict()[layer]
#     elif layer ==5:
#         elementDict = model.state_dict()
#         newLayer = np.random.random_sample(1048576)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(512, 128, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict)
#         return model.state_dict()[layer]
#     elif layer ==6:
#         elementDict = model.state_dict()
#         newLayer = np.random.random_sample(131072)
#         layerQuantile = np.quantile(newLayer, percentage)
#         binaryAns = newLayer < layerQuantile
#         newLayer[binaryAns] = 0
#         sparseTensor = newLayer.reshape(128, 64, 4, 4)
#         elementDict[layer] = torch.from_numpy(sparseTensor).float().to(device=cuda)
#         model.load_state_dict(elementDict)
#         return model.state_dict()[layer]


def runLSUN(batchSize, layer, sparcity):

    if layer == 1:
        net = pgan(1, False, 512, 512, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        print("execute")
        image = net(inp)
        
    if layer == 2:
        net = pgan(2,False, 512, 512, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        print("execute")
        image = net(inp)

    if layer == 3:
        net = pgan(3,False, 512, 512, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        # print(inp.shape)
        print("execute")
        image = net(inp)

    if layer == 4:
        net = pgan(4,False, 512, 256, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        print("execute")
        image = net(inp)

    if layer == 5:
        net = pgan(5,False, 256, 128, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        print("execute")
        image = net(inp)

    if layer == 6:
        net = pgan(6,False, 128, 64, 4)
        # layerWeightName = f"layer{layer}.weight"
        # print(layerWeightName)
        # setToZero(net,layerWeightName, sparcity)
        setToZero(net, layer, sparcity)
        inp = torch.rand(batchSize, 512,4,4)
        print("execute")
        image = net(inp)
    
    state_dict = net.state_dict()
    print(state_dict[f"layer{layer}.weight"])


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
    print(f"iteration:{iterrr} lsun layer:{layer}, sparcity:{temp}")

    runLSUN(1, layer, temp)

if __name__ == '__main__':
    main()