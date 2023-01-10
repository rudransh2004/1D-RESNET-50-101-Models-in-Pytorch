import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 


class residual_block(nn.Module):
    def __init__(self,in_channels,intermediate_channels,identity_downsample=None,stride=1):
        super().__init__()
        self.expansion =4 #This is only used to match the dimesions after applying convolution operations.
        #Since 2nd Convolution operation would change the dimentions as it has stride 1  and kernel equal to 3 

        self.conv_layer_1 = nn.Conv1d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        #After this operation we use expansion value in the next layer 
        #in order to match the dimensions 
        self.conv_layer_2 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias = False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        
        self.conv_layer_3 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels *self.expansion,
            kernel_size=1,
            stride = 1,
            padding=0,
            bias=False
        )
        self.bn3 =nn.BatchNorm1d(intermediate_channels*self.expansion)
        
        self.relu = nn.ReLU()
        
        self.identity_downsample = identity_downsample
        
        self.stride  = stride
    
    def forward(self,x):
        identity = x.clone()
        
        x = self.conv_layer_1(x)
        
        x = self.bn1(x)
        
        x = self.relu(x)
        
        x = self.conv_layer_2(x)
        
        x = self.bn2(x)
        
        x = self.relu(x)
        
        x  = self.conv_layer_3(x)
        
        x  = self.bn3(x)
        
        if self.identity_downsample is not None:
            
            identity = self.identity_downsample(identity)
        
        x += identity
        
        x = self.relu(x)
        
        return x

class Resnet(nn.Module):
    def __init__(self,layers,data_channels,num_classes):
        super(Resnet,self).__init__()
        self.in_channels =64 
        self.conv1  = nn.Conv1d(data_channels,64,stride=2,kernel_size=7,padding=3,bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._build_layer(
            layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._build_layer(
            layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._build_layer(
             layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._build_layer(
           layers[3], intermediate_channels=512, stride=2
        )
       
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        x = self.fc(x)
        return x
    
    def _build_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(intermediate_channels * 4),
            )
        layers.append(
            residual_block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(residual_block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
