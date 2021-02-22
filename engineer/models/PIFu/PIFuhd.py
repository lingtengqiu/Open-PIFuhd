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
from ..common.net_utils import freeze
import torch.nn.functional as F
from collections import OrderedDict
from engineer.models.builder import build_model,build_backbone,build_head

import logging
logger = logging.getLogger('logger.trainer')


@PIFU.register_module
class PIFuhdNet(_BasePIFuNet):
    def __init__(self,backbone,head,global_net,projection_mode:str='orthogonal',error_term:str='mse',num_views:int=1,skip_hourglass:bool=False,is_train_full_pifuhd=False):
        '''
        Parameters:
            backbone: networks to extract image features, default Hourglass
            head: networks to predict value, in PIfu, which is terms of iso-surface
            global_net: global net to extract image features
            projection_mode: how to render your 3d model to images, default orthogonal matrix
            error_term: target function or energy function, initially MSE loss
            num_view: how many images from which you want to reconstruct your model
        '''
        super(PIFuhdNet,self).__init__(projection_mode,error_term)
        try:
            assert backbone is not None
            assert head is not None
            assert global_net is not None
        except:
            raise EnvironmentError("we should provide backbone,head and depth network")

        self.backbone = build_backbone(backbone)


        
        self.head = build_head(head)

        self.global_net = build_model(global_net)
        self.is_train_full_pifuhd = is_train_full_pifuhd
        


        self.input_para =dict(
            backbone=self.backbone.name,
            head = self.head.name,
            depth = self.global_net.name,
            projection_mode= projection_mode,
            error_term= error_term,
            num_view=num_views,
            skip_hourglass=skip_hourglass,
            is_train_full_pifuhd = is_train_full_pifuhd
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

        #load coarse-pifu weight
        logger.info("resume coarse-pifu from {}".format(global_net.pretrain_weights))
        checkpoints = torch.load(global_net.pretrain_weights)
        try:
            self.global_net.load_state_dict(checkpoints['model_state_dict'])
        except:
            model_state_dict = checkpoints['model_state_dict']
            weight = OrderedDict()
            for key in model_state_dict.keys():
                weight[key.replace('module.',"")] = model_state_dict[key]
            try:
                self.global_net.load_state_dict(weight)
            except:
                self.global_net.module.load_state_dict(weight)
        
        if not self.is_train_full_pifuhd:
            logger.info("freeze coarse PIFu!")
            freeze(self.global_net)
            

    def train(self, mode=True):
        '''Sets the module in training mode.
        '''      
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.is_train_full_pifuhd:
            self.global_net.eval()
        return self



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
    def query(self, crop_query_points, local_features,labels=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        r'''Only for fine PIFuhd
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.

        Parameters:
            crop_query_points: [B, 2, N] crop query points according to crop window_size
            local_features: [B, C_fea, N] query points features from local features
            labels: gt of queried points
        Return: 
            [B, Res, N] predictions for each point
        '''

        if labels is not None:
            self.labels = labels
        #intial, we define project matrix is orthogonal
        xy = crop_query_points[:, :2, :]


        #Flag to identify, whether points inside or outside images.
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)



        if self.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), local_features]

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

    def forward(self, images, points, calibs, crop_imgs,crop_points_query,transforms=None, labels=None):
        '''
        Parameters:
            points: [B, 3, N] world space coordinates of points
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
            crop_imgs: [B, C, H, W] crop images
            crop_points_query: [B, 2, N] world space coordinate of points which maps into crop_size
        Return: 
            [B, Res, N] predictions for each point
        '''
        # Phase 1: Get local image feature and query points_features
        if not self.is_train_full_pifuhd:
            with torch.no_grad():
                self.global_net.extract_features(images)
                self.global_net.query(points=points, calibs=calibs, transforms=transforms, labels=labels)
                local_features = self.global_net.get_merge_feature()
        else:
            raise NotImplementedError
        
        # Phase 2: Get global image feature
        self.extract_features(crop_imgs)
        
        # Phase 3: point query
        self.query(crop_points_query, local_features,labels=labels)

        # get the prediction
        res = self.get_preds()
    
        # get the error
        error = self.get_error()
        
        return res, error


 
        

