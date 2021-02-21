import numpy as np
"----------------------------- Model options -----------------------------"
model = dict(
    PIFu=dict(
        type='PIFUNet', 
        head =dict(
        type='SurfaceHead',filter_channels=[257, 1024, 512, 256, 128, 1], num_views = 1, no_residual=True, last_op='sigmoid'),
        backbone=dict(
        type = 'Hourglass',num_stacks= 4,num_hourglass=2,norm='group',hg_down='ave_pool',hourglass_dim= 256),
        depth=dict(
        type='DepthNormalizer',input_size = 512,z_size=200.0),
        projection_mode='orthogonal',
        error_term='mse'
    )
)
"----------------------------- Datasets options -----------------------------"
dataset_type = 'RenderPeople'
train_pipeline =    None

test_pipeline = [dict(type='resize',size=(512,512)),dict(type='ImageToTensor',keys=['img','mask']),
    dict(type='normalize',mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
data = dict(
    train=dict(
    type = "RPDataset",
    input_dir = '../Garment/render_people_gen_train/',
    is_train = True,
    pipeline=train_pipeline,
    cache="../Garment/cache/rp_train/",
    random_multiview=False,
    img_size = 512,
    num_views = 1,
    num_sample_points = 5000, 
    num_sample_color = 0,
    sample_sigma=5.,
    check_occ='trimesh',
    debug=False
    ),
    test=dict(
    type = "Carton_Dataset",
    input_dir = '../animals/',
    is_train = False,
    pipeline=test_pipeline,
    cache="../Garment/cache/rp_test/",
    random_multiview=False,
    img_size = 512,
    num_views = 1,
    num_sample_points = 5000, 
    num_sample_color = 0,
    sample_sigma=5.,
    check_occ='trimesh',
    debug=False
    )
)
train_collect_fn = 'train_loader_collate_fn'
test_collect_fn = 'carton_test_loader_collate_fn'
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
lr_warm_up = 1e-5
warm_epoch= 5
LR=1e-3
num_epoch=100
batch_size = 4
test_batch_size = 1
scheduler=dict(
    gamma = 0.1,
    stone = [60,80] 
)
save_fre_epoch = 2
"----------------------------- evaluation setting -------------------------------"
val_epoch = 1
start_val_epoch = 6
"----------------------------- inference setting -------------------------------"
resolution = 256 #for inference
"-------------------------------- config---name   -------------------------------"
name='PIFu_Animals_HG'