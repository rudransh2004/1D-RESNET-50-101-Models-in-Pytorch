
# Project Title

This Repo will help you to make a your own 1 dimension Resnet-50 model in Pytorch from scratch. 


# Dataset Used 

Crema-d Audio classification dataset was used to test out the Resnet model. Refer to Jupyter Notebook 
to see the usage of the implemented model.




## Usage/Examples


Parameters

data_channels -> Dimentions of the input data I have used 40 since I'm using MFCC features 
                 for audio

num_classes ->  In this example it is 6 but you can change according to the no of classes
                you have in your data. 
                 

```python
from resnet import Resnet

#1D Resnet 50 model 

def Resnet50(layers = [3,4,6,3],data_channels=40, num_classes=6):
    return Resnet(layers, data_channels, num_classes)


#1D Resnet 101 model 
def ResNet101(layers= [3,4,23,3], data_channels=40 ,num_classes=6):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

#1D Resnet 152 model 
def Resnet152(layers = [3,8,36,3],data_channels=40,num_classes=6):
    return Resnet(layers,data_channels,num_classes)


```


## Acknowledgements

 - [Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/CNN_architectures)
 
## Authors

- [Rudransh Agnihtori](www.linkedin.com/in/rudransh-agnihotri-b8b102185)

