import torch
import torch.nn as nn
import torch.nn.functional as F
from engineer.utils.geometry import index, orthogonal, perspective
from torch.nn import init

class _BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode:str='orthogonal',
                 error_term:str='mse',
                 ):
        '''
        Parameters
            backbone: Which backbone you use in your PIFu model {Res32|Hourglass ....}
            head: Which function you want to learn, default: iso-surface--->surface_classifier
            depth: The network aims at predict depth of 3-D points
            projection_model : Either orthogonal or perspective.
            param error_term:  nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        '''

        super(_BasePIFuNet, self).__init__()
        self.__name = 'basePIFu'

        self.error_term = error_term
        
        if error_term == 'mse':
            self.error_term = nn.MSELoss()
        elif error_term == 'bce':
            self.error_term = nn.BCELoss()
        else:
            raise NotImplementedError

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

    def forward(self, points, images, calibs, transforms=None)->torch.Tensor:
        '''
        Parameters:
            points: [B, 3, N] world space coordinates of points
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
        Return: 
            [B, Res, N] predictions for each point
        '''

        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()


    def extract_features(self, images):
        '''
        Filter the input images
        store all intermediate features.

        Parameters:
            images: [B, C, H, W] input images
        '''
        raise NotImplementedError

    def query(self, points, calibs, transforms=None, labels=None):
        '''
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
        None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds

    def get_error(self):
        '''
        Get the network loss from the last query

        return: 
            loss term
        '''
        return self.error_term(self.preds, self.labels)



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


    @staticmethod
    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

    @staticmethod
    def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
        '''
        Initialize a network:
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
        Return:
            None
        '''
        __class__.init_weights(net, init_type, init_gain=init_gain)