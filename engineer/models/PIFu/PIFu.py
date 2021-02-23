'''
author: LingtengQiu
data: 2021-1-17
PIFu based on OpenMMlab version
'''
import torch.nn as nn
import torch
from engineer.models.registry import PIFU
from .BasePIFu import _BasePIFuNet
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F

from engineer.models.builder import build_backbone,build_depth,build_head



@PIFU.register_module
class PIFUNet(_BasePIFuNet):
    def __init__(self,backbone,head,depth,projection_mode:str='orthogonal',error_term:str='mse',num_views:int=1,skip_hourglass:bool=False):
        '''
        Parameters:
            backbone: networks to extract image features, default Hourglass
            head: networks to predict value, in PIfu, which is terms of iso-surface
            depth: networks to normalize depth of camera coordinate
            projection_mode: how to render your 3d model to images, default orthogonal matrix
            error_term: target function or energy function, initially MSE loss
            num_view: how many images from which you want to reconstruct your model
        '''
        super(PIFUNet,self).__init__(projection_mode,error_term)
        try:
            assert backbone is not None
            assert head is not None
            assert depth is not None
        except:
            raise EnvironmentError("we should provide backbone,head and depth network")
        

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.depth = build_depth(depth)

        
        self.input_para =dict(
            backbone=self.backbone.name,
            head = self.head.name,
            depth = self.depth.name,
            projection_mode= projection_mode,
            error_term= error_term,
            num_view=num_views,
            skip_hourglass=skip_hourglass
        )
        backbone_style = ""
        if 'Hourglass' in self.input_para['backbone']:
            backbone_style = "HG"
        elif 'Res' in self.input_para['backbone']:
            backbone_style = 'Res34'
        else:
            raise NotImplementedError
        self.name ="PIFu{}Net".format(backbone_style)

        #backbone
        self.num_views = num_views
        self.skip_hourglass = skip_hourglass

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []
        
        self.init_net(self)


    def extract_features(self, images:torch.Tensor):
        #Tensor->torch.Tensor
        r'''
        extract the input images
        store all intermediate features.

        Parameters:
            images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.backbone(images)
        # If it is not in training, only produce the last im_feat

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
    def query(self, points, calibs, transforms=None, labels=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        r'''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.

        Parameters:
            points: [B, 3, N] world space coordinates of points
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
            labels: Optional [B, Res, N] gt labeling
        Return: 
            [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        #project, points into camera coordinate
        xyz = self.projection(points, calibs, transforms)

        #intial, we define project matrix is orthogonal
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        #Flag to identify, whether points inside or outside images.
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)


        z_feat = self.depth(z, calibs=calibs)

        if self.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if self.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0

            pred, phi = self.head(point_local_feat)
            pred = in_img[:,None].float() *pred
            self.intermediate_preds_list.append(pred)
        #merge_layer for pifuhd
        self.phi = phi

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        #type ->torch.Tensor
        r'''
        Get the image features
        
        return: 
            [B, C_feat, H, W] image feature after backbone
        '''
        return self.im_feat_list[-1]

    def get_merge_feature(self):
        '''
        Get the merge_feature according to query points

        return:
            [B, C_feat, N]
        '''
        return self.phi

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        Therefore the loss function is supervised in all stacks of Hourglass
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, points, calibs, transforms=None, labels=None):
        '''
        Parameters:
            points: [B, 3, N] world space coordinates of points
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
        Return: 
            [B, Res, N] predictions for each point
        '''
        # Get image feature
        self.extract_features(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()
    
        # get the error
        error = self.get_error()
        
        return res, error


 
        

