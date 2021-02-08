'''
Author:lingteng Qiu
data: 2021-1-25
'''
from ..registry import PIPELINES
import torch
import numpy as np
import cv2
import numbers
import random
import torch.nn.functional as F




@PIPELINES.register_module
class img_pad(object):
    def __init__(self,pad_ratio = 0.1):
        self.pad_ratio = pad_ratio
    def __call__(self,data):
        img = data['img']
        mask = data['mask']

        front = data['front_normal']
        back = data['back_normal']
        back_mask = data['back_mask']


        size = img.shape[0]
        h,w,_ = img.shape
        pad_size = int(size*self.pad_ratio)


        img = np.pad(img,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values = 0)

        mask = np.pad(mask,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values = 0)

        if front is not None:
            data['front_normal'] = np.pad(front,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values = 0)
            data['back_normal'] = np.pad(back,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values = 0)
            data['back_mask'] = np.pad(back_mask,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values = 0)          


        
        data['mask'] = mask
        data['img'] = img
        data['th'] = h 
        data['tw'] = w
        return data

    def __repr__(self):
        repr_str="{}(pad_ratio={})".format(self.__class__.__name__,self.pad_ratio)
        return repr_str



@PIPELINES.register_module
class flip(object):
    def __init__(self,flip_ratio = 0.5):
        self.flip_ratio = flip_ratio
    def __call__(self,data):
        img = data['img']
        mask = data['mask']
        
        front = data['front_normal']
        back = data['back_normal']
        back_mask = data['back_mask']

        scale_intrinsic = data['scale_intrinsic']
        if np.random.rand() > 1 - self.flip_ratio:
            scale_intrinsic[0, 0] *= -1
            img = img[:,::-1,:]
            mask = mask[:,::-1,:]
            if front is not None:
                front = front[:,::-1,:]
                back = back[:,::-1,:]
                back_mask = back_mask[:,::-1,:]
                data['flip'] = True

        data['mask'] = mask
        data['img'] = img
        data['scale_intrinsic']=scale_intrinsic
        data['front_normal'] = front
        data['back_normal'] = back
        data['back_mask'] = back_mask



        return data

    def __repr__(self):
        repr_str="{}(flip_ratio={})".format(self.__class__.__name__,self.flip_ratio)
        return repr_str

@PIPELINES.register_module
class scale(object):
    def __init__(self):
        pass
    def __call__(self,data):
        rand_scale = np.random.uniform(0.9, 1.1)
        img = data['img']
        mask = data['mask']
        scale_intrinsic = data['scale_intrinsic']

        front = data['front_normal']
        back = data['back_normal']

        back_mask = data['back_mask']


        h,w,_ = img.shape

        w = int(rand_scale * w)
        h = int(rand_scale * h)

        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        mask  =cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)

        if front is not None:        
            data['front_normal'] = cv2.resize(front,(w,h),interpolation=cv2.INTER_LINEAR)
            data['back_normal'] = cv2.resize(back,(w,h),interpolation=cv2.INTER_LINEAR)
            data['back_mask'] = cv2.resize(back_mask,(w,h),interpolation=cv2.INTER_NEAREST)

        scale_intrinsic *= rand_scale
        scale_intrinsic[3, 3] = 1
        
        data['mask'] = mask
        data['img'] = img
        data['scale_intrinsic']=scale_intrinsic


        return data

    def __repr__(self):
        repr_str="{}()".format(self.__class__.__name__)
        return repr_str


@PIPELINES.register_module
class random_crop_trans(object):
    def __init__(self,enable=True):
        self.enable = enable
    def __call__(self,data):

        img = data['img']
        mask = data['mask']
        h,w,_ = img.shape
        th,tw = data['th'],data['tw']
        trans_intrinsic = data['trans_intrinsic']


        front = data['front_normal']
        back = data['back_normal']
        back_mask = data['back_mask']


        if self.enable:
            # make sure that dx,dy cannot exceed img_size
            # this is very important
            dx = np.random.randint(-int(round((w - tw) / 10.)),
                                int(round((w - tw) / 10.)))
            dy = np.random.randint(-int(round((h - th) / 10.)),
                                int(round((h - th) / 10.)))
        else:
            dx = 0
            dy = 0
        
        trans_intrinsic[0, 3] = -dx / float(tw // 2)
        trans_intrinsic[1, 3] = -dy / float(th // 2)


        x1 = int(round((w - tw) / 2.)) + dx
        y1 = int(round((h - th) / 2.)) + dy
        
        #crop
        data['img'] = img[y1:y1+th,x1:x1+tw,...]
        data['mask'] = mask[y1:y1+th,x1:x1+tw,...]
        data['trans_intrinsic'] = trans_intrinsic

        if front is not None:        
            data['front_normal'] = front[y1:y1+th,x1:x1+tw,...]
            data['back_normal'] = back[y1:y1+th,x1:x1+tw,...]
            data['back_mask'] = back_mask[y1:y1+th,x1:x1+tw,...]

        return data

    def __repr__(self):
        repr_str="{}(enable={})".format(self.__class__.__name__,self.enable)
        return repr_str

@PIPELINES.register_module
class resize(object):
    def __init__(self,size:tuple,normal=False):
        self.size = size
        self.normal = normal
    def __call__(self,data):
        img = data['img']
        mask = data['mask']
        data['img'] = cv2.resize(img,self.size)
        data['mask'] = cv2.resize(mask,self.size,interpolation=cv2.INTER_NEAREST)

        if self.normal:
            data['front_normal'] = cv2.resize(data['front_normal'],self.size)
            data['back_normal'] = cv2.resize(data['back_normal'],self.size)
            data['back_mask'] = cv2.resize(data['back_mask'],self.size,interpolation=cv2.INTER_NEAREST)


        # cv2.namedWindow('img', 0);
        # cv2.resizeWindow('img', 1024,1024);
        # cat = np.concatenate([data['back_normal'],data['front_normal']],axis=1)
        # cv2.imshow("img",cat)
        # cv2.waitKey()

        return data

    def __repr__(self):
        repr_str="{}(size={}, normal={})".format(self.__class__.__name__,self.size, self.normal)
        return repr_str

@PIPELINES.register_module
class to_camera(object):
    def __init__(self):
        pass
    def __call__(self,data):
        uv_intrinsic= data['uv_intrinsic']
        scale_intrinsic= data['scale_intrinsic']
        trans_intrinsic= data['trans_intrinsic']
        extrinsic= data['extrinsic']
        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = np.matmul(intrinsic, extrinsic)

        data['intrinsic'] = intrinsic
        data['calib'] = calib
        return data

    def __repr__(self):
        repr_str="{}()".format(self.__class__.__name__)
        return repr_str


@PIPELINES.register_module
class normalize_normal(object):
    def __init__(self):
        pass
    def __call__(self,data):
        front = data['front_normal']
        back = data['back_normal']
        front = (front)*2-1
        back = (back)*2-1
        if data['flip']:
            #BGR
            #we fine actually, we should
            front[2,...] = - front[2,...]
            back[2,...] = - back[2,...]

        data['front_normal'] = front
        data['back_normal'] = back   
        return data
    def __repr__(self):
        repr_str="{}()".format(self.__class__.__name__)
        return repr_str

@PIPELINES.register_module
class normalize(object):
    def __init__(self,mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]):
        #std: bgr
        self.mean = mean
        self.std=std
    def __call__(self,data):
        img = data['img']
        mean =torch.Tensor(self.mean)[:,None,None].expand_as(img)
        std = torch.Tensor(self.std)[:,None,None].expand_as(img)
        
        img = (img - mean)/std
        data['img'] = img
        return data

    def __repr__(self):
        repr_str="{}(mean={},std={})".format(self.__class__.__name__,self.mean,self.std)
        return repr_str

@PIPELINES.register_module
class color_jitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,keys=None):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.keys = keys

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_brightness(self,img,brightness_factor):
        #_blend in pytorch mean = 0
        img = img*brightness_factor
        return img
    def adjust_contrast(self,img,contrast_factor):
        # _blend in pytorch, mean = gray.mean
        mean = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).mean()
        return img*(contrast_factor)+(1-contrast_factor)*mean
    def adjust_saturation(self,img,saturation_factor):
        # _blend in pytorch, mean = gray_image
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img2 = img2[...,None]
        return img*(saturation_factor)+(1-saturation_factor)*img2
    def adjust_hue(self,img,hue_factor):
        """Adjust hue of an image.

        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.

        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.

        See `Hue`_ for more details.

        .. _Hue: https://en.wikipedia.org/wiki/Hue

        Args:
            img (CV2): CV2 Image to be adjusted.
            hue_factor (float):  How much to shift the hue channel. Should be in
                [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                HSV space in positive and negative direction respectively.
                0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                with complementary colors while 0 gives the original image.

        Returns:
            CV2 Image: Hue adjusted image.
        """
        if not(-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
        #only 180 due to the limit of bits
        h, s, v = np.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),3,axis=2)
        np_h = np.asarray(h,dtype = np.int16)
        with np.errstate(over='ignore'):
            np_h += np.int16(hue_factor * 180)
        np_h = np.mod(np_h,180)
        return cv2.cvtColor(np.concatenate([h,s,v],axis = 2),cv2.COLOR_HSV2BGR)
         
    def forward(self,img):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = np.random.uniform(brightness[0],brightness[1])
                img = self.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = np.random.uniform(contrast[0],contrast[1])
                img = self.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = np.random.uniform(saturation[0],saturation[1])
                img = self.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = np.random.uniform(hue[0],hue[1])
                img = self.adjust_hue(img, hue_factor)
        img = np.clip(img,0,255)
        return np.asarray(img,dtype=np.uint8)

    def __call__(self,data):
        for key in self.keys:
            _data = data[key]
            data[key] = self.forward(_data)
        return data

    def __repr__(self):
        repr_str="{}(brightness={}, contrast={}, saturation={}, hue={})".format(self.__class__.__name__,self.brightness,self.contrast,self.saturation,self.hue)
        return repr_str