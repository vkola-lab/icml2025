'''
ResNet in PyTorch.

This implementation is based on kuangliu's code
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and the PyTorch reference implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch.nn as nn
import torch.nn.functional as F
def make_layer(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        if stride >= 1:
            layers.append(block(in_planes, planes, stride))
        else:
            layers.append(UpsamplingBlock(in_planes, planes))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Input conv.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks.
        channels = 64
        stride = 1
        blocks = []
        for num in num_blocks:
            blocks.append(self._make_layer(block, channels, num, stride=stride))
            channels *= 2
            stride = 2
        self.layers = nn.ModuleList(blocks)
        
        
        
        self.fea_layer = nn.Sequential(
        # Upsampling block.
        # make_layer(BasicBlock, in_planes=256, planes=128, num_blocks=2, stride=0.5),
        make_layer(BasicBlock, in_planes=512, planes=256, num_blocks=2, stride=0.5),
        
        # Output selections.
        # nn.Conv2d(128, 1, 1)
        #nn.Conv2d(256, 8, 1) # 8 for dim=512
        nn.Conv2d(256, 8, 1) # 4 for dim=256
        )
        
        # Output layer.
        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(512*block.expansion, num_classes)
            #self.linear2 = nn.Linear(512*block.expansion, 64)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, fea_return=False):
        # Input conv.
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks.
        for layer in self.layers:
            out = layer(out)

        # Output layer.
        if self.num_classes is not None:
            #if fea_return:
            #    fea = self.fea_layer(out)
            #    fea = fea.view(fea.size(0), -1)
                #print(fea.shape)
            out = F.avg_pool2d(out, 4)
            fea = out.view(out.size(0), -1)
            #out2 = self.linear(out)
            out = self.linear(fea)
            #print('Feature Shape')
            #print(fea.shape)
        if fea_return:
            return out, fea
        else:
            return out

def ResNet10(num_classes, in_channels=3):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, in_channels)

def ResNet18(num_classes, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)

class UpsamplingBlock(nn.Module):
    '''Custom residual block for performing upsampling.'''
    expansion = 1
    
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                        in_planes, self.expansion*planes, kernel_size=2,
                        stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

def ResNet18class(num_classes=10):
    '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
    return nn.Sequential(
        # Initial conv-bn-relu sequence.
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1),

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2),
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2),
        
        # Block 4.
        make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2),

        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

class ResNet18class(nn.Module):
    def __init__(self,num_classes=10): 
        super(ResNet18class, self).__init__()
        '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
        
        # Initial conv-bn-relu sequence.
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU()
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        self.block1=make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1)

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        self.block2=make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2)
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        self.block3=make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2)
        
        # Block 4.
        self.block4=make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2)

        self.avgp = nn.AvgPool2d(4)
        self.flat = nn.Flatten()
        self.lin  = nn.Linear(512, num_classes)
    def forward(self, x, fea_return=False):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        x = self.block4(x)
        x = self.lin(self.flat(self.avgp(x)))
        return x
        


def ResNet18Backbone():
    '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
    return nn.Sequential(
        # Initial conv-bn-relu sequence.
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1),

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2),
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2),
        
        # Block 4.
        make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2)
        
    )

def ResNet18ClassifierHead(num_classes=10):
    '''ResNet18 classifier head, including residual block, GAP and FC layer.'''
    return nn.Sequential(
        # # Block 4.
        # make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=2, stride=2),
        
        # GAP + FC.
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

def ResNet18SelectorHead():
    '''Custom upsampling module to select 2x2 patches (assuming 32x32 input).'''
    return nn.Sequential(
        # Upsampling block.
        # make_layer(BasicBlock, in_planes=256, planes=128, num_blocks=2, stride=0.5),
        make_layer(BasicBlock, in_planes=512, planes=256, num_blocks=2, stride=0.5),
        
        # Output selections.
        # nn.Conv2d(128, 1, 1)
        nn.Conv2d(256, 1, 1)
    )

class medmnistCNN(nn.Module):
    def __init__(self,in_channels=3,num_classes=9,multiplier=1):      
        super(medmnistCNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32*multiplier, out_channels=64*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64*multiplier, out_channels=64*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        self.avg_conv = nn.Conv2d(64*multiplier, 16, kernel_size=4, stride=2, padding=0) # diff2
        #self.avg_conv = nn.Conv2d(64*multiplier, 4, kernel_size=1, stride=1, padding=0)
        #self.convt1 = nn.ConvTranspose2d(in_channels=64*multiplier, out_channels=32*multiplier, kernel_size=4, stride=2, padding=1)
        #self.avg_conv = nn.Conv2d(in_channels=32*multiplier, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.avg = nn.AvgPool2d(4)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(64*multiplier, num_classes)
        
    def forward(self, x, fea_return=False):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        fea = self.relu4(self.conv4(x))

        x=self.avg(fea)
        x=self.flat(x)
        x=self.linear(x)
        if fea_return == True:
            #print(fea.shape)
            #fea2 = self.avg_conv2(fea) 
            fea = self.avg_conv(fea) # (diff2-diff3-diff4)

            #diff5
            #fea = self.convt1(fea)
            #fea = self.avg_conv(fea)
            
            return x,fea
        else:
            return x

class medmnistCNN_value(nn.Module):
    def __init__(self,in_channels=3,num_classes=9, multiplier=1):      
        super(medmnistCNN_value, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32*multiplier, out_channels=64*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64*multiplier, out_channels=64*multiplier, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        
        

        self.convt1 = nn.ConvTranspose2d(in_channels=64*multiplier, out_channels=32*multiplier, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32*multiplier, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x, fea_return=False):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        x = self.convt1(x)
        x = self.conv5(x)
        
        return x


class resnet18_value(nn.Module):
    def __init__(self,in_channels=3,num_classes=9): 
        super(resnet18_value, self).__init__()
        '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
        
        # Initial conv-bn-relu sequence.
        self.conv1=nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU()
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        self.block1=make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1)

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        self.block2=make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2)
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        self.block3=make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2)
        
        # Block 4.
        self.block4=make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2)

        # Block 5 for value
        self.block5=make_layer(BasicBlock, in_planes=512, planes=256, num_blocks=2, stride=0.5) 
        # Output selections.
        # nn.Conv2d(128, 1, 1)
        self.conv2=nn.Conv2d(256, num_classes, 1)
    def forward(self, x, fea_return=False):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        x = self.block4(x)
        #print(x.shape)
        x = self.block5(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        #import time; time.sleep(1000)
        return x

class resnet18_value28(nn.Module):
    def __init__(self,in_channels=3,num_classes=9): 
        super(resnet18_value28, self).__init__()
        '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
        
        # Initial conv-bn-relu sequence.
        self.conv1=nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU()
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        self.block1=make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1)

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        self.block2=make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2)
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        self.block3=make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2)
        
        # Block 4.
        #self.block4=make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2)

        # Block 5 for value
        self.block5=make_layer(BasicBlock, in_planes=256, planes=256, num_blocks=2, stride=0.5) 
        # Output selections.
        # nn.Conv2d(128, 1, 1)
        self.conv2=nn.Conv2d(256, num_classes, 1)
    def forward(self, x, fea_return=False):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        #x = self.block4(x)
        #print(x.shape)
        x = self.block5(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        #import time; time.sleep(1000)
        return x       

class CNNBackbone(nn.Module):
    def __init__(self):      
        super(CNNBackbone, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        return x



def CNNClassifierHead(num_classes=9):
    '''ResNet18 classifier head, including residual block, GAP and FC layer.'''
    return nn.Sequential(
        # # Block 4.
        # make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=2, stride=2),
        
        # GAP + FC.
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(64, num_classes)
    )

def CNNSelectorHead():
    '''Custom upsampling module to select 2x2 patches (assuming 32x32 input).'''
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
    )