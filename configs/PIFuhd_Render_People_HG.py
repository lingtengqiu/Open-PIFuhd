'''
training coarse PIFu with gt normal map
'''
import numpy as np
"---------------------------- normal options -----------------------------"
use_front = True
use_back = True
"----------------------------- Model options -----------------------------"
model = dict(
    PIFu=dict(
        type='PIFUNet', 
        head =dict(
        type='PIFuhd_Surface_Head',filter_channels=[257, 1024, 512, 256, 128, 1], merge_layer = 2, res_layers=[2, 3, 4], norm= None,last_op='sigmoid'),
        backbone=dict(
        type = 'Hourglass',num_stacks= 4,num_hourglass=2,norm='group',hg_down='ave_pool',hourglass_dim= 256,use_front = use_front, use_back = use_back),
        depth=dict(
        type='DepthNormalizer',input_size = 512,z_size=200.0),
        projection_mode='orthogonal',
        error_term='mse'
    )
)
"----------------------------- Datasets options -----------------------------"
dataset_type = 'RenderPeople'
train_pipeline = [
    dict(type='img_pad'),dict(type='flip',flip_ratio=0.5),dict(type='scale'),dict(type='random_crop_trans'), 
    dict(type='color_jitter',brightness=0., contrast=0., saturation=0.0, hue=0.0,keys=['img']),dict(type='resize',size=(512,512),normal=True),
    dict(type='to_camera'),dict(type='ImageToTensor',keys=['img','mask','front_normal','back_normal','back_mask']),dict(type='normalize',mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ,dict(type='normalize_normal'),dict(type='ToTensor',keys=['calib','extrinsic']),
]

test_pipeline = [
    dict(type='resize',size=(512,512),normal =True),
    dict(type='to_camera'),
    dict(type='ImageToTensor',keys=['img','mask','front_normal','back_normal','back_mask']),
    dict(type='normalize',mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    dict(type='normalize_normal'),
    dict(type='ToTensor',keys=['calib','extrinsic']),
]
data = dict(
    train=dict(
    type = "RPDataset",
    input_dir = '../Garment/render_gen_1024_train/',
    is_train = True,
    pipeline=train_pipeline,
    cache="../Garment/cache/render_gen_1024/rp_train/",
    random_multiview=False,
    img_size = 1024,
    num_views = 1,
    num_sample_points = 5000, 
    num_sample_color = 0,
    sample_sigma=[5.,3.],
    check_occ='trimesh',
    debug=False,
    span=1,
    normal = True,
    sample_aim = 5.
    ),

    test=dict(
    type = "RPDataset",
    input_dir = '../Garment/render_gen_1024_test/',
    is_train = False,
    pipeline=test_pipeline,
    cache="../Garment/cache/render_gen_1024/rp_test/",
    random_multiview=False,
    img_size = 1024,
    num_views = 1,
    num_sample_points = 5000, 
    num_sample_color = 0,
    sample_sigma=[5.,3.],
    check_occ='trimesh',
    debug=False,
    span = 90,
    normal = True,
    sample_aim = 5.
    )
)
train_collect_fn = 'train_loader_collate_fn'
test_collect_fn = 'test_loader_collate_fn'
"----------------------------- checkpoints options -----------------------------"
checkpoints = "./checkpoints"
logger = True
"----------------------------- optimizer options -----------------------------"
optim_para=dict(
    optimizer = dict(type='RMSprop',lr=1e-3,momentum=0, weight_decay=0.0000),
)
"----------------------------- training strategies -----------------------------"
num_gpu = 1
lr_policy="stoneLR"
lr_warm_up = 1e-4
warm_epoch= 1
LR=1e-3
num_epoch=12
batch_size = 4
test_batch_size = 1
scheduler=dict(
    gamma = 0.1,
    stone = [10] 
)
save_fre_epoch = 1
"----------------------------- evaluation setting -------------------------------"
val_epoch = 1
start_val_epoch = 0
"----------------------------- inference setting -------------------------------"
resolution = 256 #for inference
"-------------------------------- config name --------------------------------"
name='PIFuhd_Render_People_HG'

"-------------------------------- render --------------------------------"
render_cfg = dict(
    type='Noraml_Render',
    width = 1024,
    height = 1024,
    render_lib ='face3d',
    flip =True
)