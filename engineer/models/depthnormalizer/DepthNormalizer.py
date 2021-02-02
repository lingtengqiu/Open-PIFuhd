import torch.nn as nn
import torch
from engineer.models.registry import DEPTH
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F

@DEPTH.register_module
class DepthNormalizer(nn.Module):
    def __init__(self, input_size:int = 512,z_size:int = 200):
        '''
        Class about DepthNormalizer
        which use to generate depth-information
        Parameters:
            input_size: the size of image, initially, 512 x 512
            z_size:     z normalization factor
        '''

        super(DepthNormalizer, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.name = "DepthNormalizer"
        self.input_para=dict(
            input_size=input_size,
            z_size=z_size
        )

    def forward(self, z, calibs=None, index_feat=None)->torch.Tensor:
        '''
        Normalize z_feature
        Parameters:
            z_feat: [B, 1, N] depth value for z in the image coordinate system
            calibs: cameara matrix
        :return:
            normalized features
            z_feat [B,1,N]
        '''
        z_feat = z * (self.input_size // 2) / self.z_size
        return z_feat



    @property
    def name(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    @name.setter
    def name(self,v):
        self.__name = v