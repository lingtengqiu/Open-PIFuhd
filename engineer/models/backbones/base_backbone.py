import torch.nn as nn
import torch
import numpy as np


class _BaseBackbone(nn.Module):
    def __init__(self):
        super(_BaseBackbone, self).__init__()
        self.__name = 'BaseBackbone'

    @property
    def name(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    @name.setter
    def name(self,name:str):
        self.__name = name
    def set_optimizer(self,optim,lr =1e-3):
        '''
        The function is employed to set different learning rate in 
        differnet parts.
        Parameters: 
            optim: optimizer which you set from cfg
        '''
        raise NotImplementedError

        
