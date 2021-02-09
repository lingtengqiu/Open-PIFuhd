import torch.nn as nn
import torch
from engineer.models.registry import HEADS
from .base_head import _BaseHead
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F
from einops import rearrange

@HEADS.register_module
class PIFuhd_Surface_Head(_BaseHead):
    '''
    MLP: aims at learn iso-surface function Implicit function
    where 0->outside, 1-> inside
    therefore, we define 0.5 is iso-surface
    '''
    def __init__(self, 
                 filter_channels, 
                 merge_layer=0,
                 res_layers=[],
                 norm='group',
                 last_op=None):
        '''
        Parameters:
            filter_channels: List mlp layers default [257, 1024, 512, 256, 128, 1]
            merge_layer: it means which layer you want to employ in fine PIFu model
            res_layers: whether you wana employ residual block ? Default [2,3,4]
            norm: use group normalization or not
            last_op: what kind of operator you want to use to get iso-value. default sigmoid-->(0,1)
        '''
        super(PIFuhd_Surface_Head, self).__init__()
        
        self.name = 'PIFuhd_Surface_Head'
        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        else:
            raise NotImplementedError("only sigmoid function could "
                 "be used in terms of sigmoid")

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2

        self.res_layers = res_layers
        self.norm = norm

        self.input_para = dict(
            filter_channels=filter_channels,merge_layer=merge_layer,res_layers=res_layers,last_op=self.last_op
        )


        for l in range(0, len(filter_channels)-1):
            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    filter_channels[l] + filter_channels[0],
                    filter_channels[l+1],
                    1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1],
                    1))
            if l != len(filter_channels)-2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l+1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l+1]))

    def forward(self, feature)->torch.Tensor:
        '''feature may include multiple view inputs
        Parameters:
            feature: [B, C_in, N]
        return:
            prediction: [B, C_out, N] and merge layer features
        '''

        y = feature
        tmpy = feature
        phi = None

        for i, f in enumerate(self.filters):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))         
            if i == self.merge_layer:
                phi = y.clone()

        if self.last_op is not None:
            y = self.last_op(y)

        return y, phi
