'''
@author: lingteng qiu
@version:1.0
'''
from einops import rearrange
import torch
import torchvision.transforms as transforms
def train_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''

    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []


    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append( rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal)

def test_loader_collate_fn(batches):
    '''collection function of dataloder for testing
    '''

    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []


    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append( rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal)




def carton_test_loader_collate_fn(batches):
    

    '''collection function of dataloder for testing
    '''
    img_list = []
    name_list = []

    for batch in batches:
        img = batch['img']
        img_list.append(img)
        name_list.append(batch['name'])

    imgs = torch.cat(img_list,dim=0)

    return dict(name=name_list,img=imgs)