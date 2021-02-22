'''
@author:lingteng qiu
@name:Open-Pifu
'''
import torch.nn as nn
import torch
from engineer.models.registry import BACKBONES
from .base_backbone import _BaseBackbone
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F

class HourGlass_Stack(nn.Module):
    '''
    Basic Hourglass stack block. 
    '''
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass_Stack, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm
        self._generate_network(self.depth)

    def _generate_network(self, level):
        #build module through recursion
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

@BACKBONES.register_module
class Hourglass(_BaseBackbone):
    '''
    Hourglass network uses Hourglass stacks as the image filter.
    It does the following:
        Hourglass as a backbone for PIFu networks

    '''
    def __init__(self,num_stacks:int = 4,num_hourglass:int=2,norm:str ='group',hg_down:str='ave_pool',hourglass_dim : int = 256 , use_front= \
        False, use_back =False):
        '''
        Initial function of Hourglass
        Parameters:
            --num_stacks: how many Hourglass stack you use. Default = 4
            --num_hourglass, how many block of hourglass stack. Default =2
            --norm: which normalization you use? group for group normalization while
            batch for batchnormalization
            --hg_down', down sample method, type=str, default='ave_pool', help='ave pool || conv64 || conv128 || not_down'
            --hourglass_dim', the number of channels of output features of hourglass network stack
             type=int, default='256', help='256 | 512'
        '''
        super(Hourglass, self).__init__()
        self.name = 'Hourglass Backbone'
        self.num_stacks = num_stacks
        self.num_hourglass = num_hourglass
        self.norm = norm
        self.hg_down = hg_down
        self.hourglass_dim = hourglass_dim

        self.use_front = use_front
        self.use_back = use_back

        self.input_para={"num_stacks":num_stacks,'num_hourglass':num_hourglass,'norm':norm,'hg_down':hg_down,'hourglass_dim':hourglass_dim,
        "use_front": use_front,"use_back": use_back}
        
        inc = 3
        if self.use_front:
            inc+=3
        if self.use_back:
            inc+=3

        # backbone of resnet
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3)
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'ave_pool' or self.hg_down == 'no_down':
            self.conv2 = ConvBlock(64, 128, self.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)


        # Stacking part
        for hg_module in range(self.num_stacks):
            self.add_module('m' + str(hg_module), HourGlass_Stack(1, num_hourglass, 256, self.norm))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.norm))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            self.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_stacks - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.hourglass_dim,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self,x:torch.Tensor):
        '''
        Parameters:
            X: Tensor[B,3,512,512] according to PIFu
        
        Return:
            features after Hourglass backbone
        '''

        x = F.relu(self.bn1(self.conv1(x)), True)
        #B,64,256,256
        tmpx = x

        if self.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        elif self.hg_down == 'no_down':
            x = self.conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        
        normx = x
        #B, 128,128,128

        x = self.conv3(x)
        x = self.conv4(x)
        previous = x

        '''
        until now, 
        tmpx  ->[B, 64,256,256]
        normx ->[B,128,128,128]
        x     ->[B,128,128,128]
        '''

        #Hourglass Block
        outputs = []
        for i in range(self.num_stacks):
            hg = self._modules['m' + str(i)](previous)
            

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps?
            # I believe heatmap is original annotation of pose estimator
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_stacks - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, tmpx.detach(), normx



    


