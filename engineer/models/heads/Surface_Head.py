import torch.nn as nn
import torch
from engineer.models.registry import HEADS
from .base_head import _BaseHead
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F
from einops import rearrange


@HEADS.register_module
class SurfaceHead(_BaseHead):
    '''
    MLP: aims at learn iso-surface function Implicit function
    where 0->outside, 1-> inside
    therefore, we define 0.5 is iso-surface
    '''
    def __init__(self, filter_channels:list, num_views:int=1, no_residual:bool=False, last_op='sigmoid'):
        '''
        Parameters:
            filter_channels: List mlp layers default [257, 1024, 512, 256, 128, 1]
            num_views: how many view you want to reconstruction from rgb images
            no_residual: whether you wana employ residual block ? Default = False
            last_op: what kind of operator you want to use to get iso-value. default sigmoid-->(0,1)
        '''
        super(SurfaceHead, self).__init__()
        
        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.filter_channels = filter_channels
        self.name = 'SurfaceHead'
        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        else:
            raise NotImplementedError("only sigmoid function could "
                 "be used in terms of sigmoid")

        self.input_para = dict(
            filter_channels=self.filter_channels,num_views=self.num_views,no_residual=self.no_residual,last_op=self.last_op
        )

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature)->torch.Tensor:
        '''
        Parameters:
            feature:  [B,  C_in, numpoints] tensors of image features

        return:
            value of implicit function of surface: [B,1,numpoints]
        '''

        y = feature
        tmpy = feature

        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = rearrange(y,'(b v) c n -> b v c n',v=self.num_views).mean(dim=1)
                tmpy = rearrange(tmpy, '(b v) c n -> b v c n',v = self.num_views).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)
        return y,tmpy