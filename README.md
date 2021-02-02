# OPEC-Net
[Peeking into occluded joints: A novel framework for crowd pose estimation](https://arxiv.org/pdf/2003.10506.pdf)(ECCV2020)  
![](show_imgs/pipeline.png)

# Dependencies  
- PyTorch(>=0.4 && <=1.1)  
- mmcv
- OpenCV
- visdom 
- pycocotools

This code is tested under Ubuntu 18.04, CUDA 10.1, cuDNN 7.1 environment with two NVIDIA 1080Ti GPUs.

Python 3.6.5 version is used for development.

# OCPose

## About OCPose
We build a new dataset, called Occluded Pose(OCPose), that includes more heavy occlusions to evaluate the MPPE. It contains challenging invisible jointsand complex intertwined human poses.


Dataset | Total | IoU>0.3 | IoU>0.5 | IoU>0.75 | Avg IoU
:--:|:--:|:--:|:--:|:--:|:--:
CrowdPose | 20000 | 8704(44%) | 2909(15%) | 309(2%) | 0.27  
COCO2017 | 118287 | 6504(5%) | 1209(1%) | 106(<1%) | 0.06  
MPII | 24987 | 0 | 0 | 0 | 0.11
OCHuman | 4473 | 3264(68%) | 3244(68%) | 1082(23%) | 0.46  
OCPose | 9000 | 8105(**90%**) | 6843(**76%**) | 2442(**27%**) | **0.47**

## Download

[Images](https://drive.google.com/file/d/1oQ1_epocYgvlha4eowt1FS-5f89XU7xw/view?usp=sharing)  
[Annotations](https://drive.google.com/file/d/1z8xlN56x9aKve4YSEudYJOJOPt4YaC7H/view?usp=sharing)


# Quick Start under CrowdPose Datasets

## Download Datasets
[COCO2017](https://cocodataset.org)  
[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)  

## Download processed annotations

pls, Download annotations processed by sampling rules according to our paper  
[train_process_datasets](https://drive.google.com/file/d/1WlZETQuOJWWARos8nnQw9XRamsTRTrrA/view?usp=sharing)  
[test_process_datasets](https://drive.google.com/file/d/1YQx0z_lVy8O1ithXp_16dtNjqJrJPvgI/view?usp=sharing)  

## Pretrain module
Here, we employ top-down module([Alphapose+](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) based on pytorch) as our initial module.  
The pretrain checkpoints trained by official codes could be download as following:  
[SPPE](https://drive.google.com/file/d/1Wcf5CWYGeMsfKn77Pu5wk6GpwEIScY2q/view?usp=sharing)  
[yolov3](https://drive.google.com/file/d/1IAtLxnOkE5DkTJ5Lsi7kLLU-edAgzaw_/view?usp=sharing)  

## Training 
Instead of using pretrain module in coco2017, we simply provide you quick-start version, where you merely train the OPEC-Net from processed data including both coco and CrowdPose.

Before training, the structure of projects like:  

```
coco
|   train2017
|      xxxxx.jpg   
crowdpose
|   images
|      xxxxx.jpg
project
│   
│     
│
└───test_process_datasets
│      download from Download processed annotations
│      
│   
└──────weights
│       │-- sppe
│       │     sppe weights
│       |   
│       |-- ssd
|       |   
|       |   
|       └───yolo 
|              yolow eights
└───train_process_datasets
       download from Download processed annotations
```
 
 ## Training script
 e.g.  
 ```
 TRAIN_BATCH_SIZE=14
 CONFIG_FILES=./configs/OPEC_GCN_GrowdPose_Test_FC.py
 bash train.sh ${TRAIN_BATCH_SIZE} ${CONFIG_FILES} 
 ```
 after training, the result of CrowdPose is save into checkpoints/name/mAP.txt  
 the format of results like:
 ```
 epoch (without best match) (use best match) 
 ```

## Test script
e.g.  
```
CHECKPOINTS_DIRS='path to your checkpoints files'
CONFIG_FILES =./configs/OPEC_GCN_GrowdPose_Test_FC.py
bash test.sh ${CHECKPOINTS_DIRS} ${CONFIG_FILES}
```

# Results

Result on CrowdPose-test:  

Method | mAP@50:95 | AP50 | AP75 | AP80 | AP90
:--:|:--:|:--:|:--:|:--:|:--:
Mask RCNN | 57.2 | 83.5 | 60.3 | - | -
Simple Pose | 60.8 | 81.4 | 65.7 | - | -
AlphaPose+ | 68.5 | 86.7 |73.2 | 66.9 | 45.9
**OPEC-Net** | **70.6**| **86.8** | **75.6** | **70.1** | **48.8**

Result on OCHuman:  

Method | mAP@50:95 | AP50 | AP75 | AP80 | AP90
:--:|:--:|:--:|:--:|:--:|:--:
AlphaPose+ | 27.5 | 40.8 |29.9 | 24.8 | 9.5
**OPEC-Net** | **29.1** | **41.3** | **31.4** | **27.0** | **12.8**

Result on OCPose:  

Method | mAP@50:95 | AP50 | AP75 | AP80 | AP90
:--:|:--:|:--:|:--:|:--:|:--:
Simple Pose | 27.1 | 54.3 | 24.2 | 16.8 | 4.7
AlphaPose+ | 30.8 | 58.4 |28.5 | 22.4 | 8.2
**OPEC-Net** | **32.8** | **60.5** | **31.1** | **24.0** | **9.2**

# Visualization
![result on crowdpose](show_imgs/visualization.png)

# Citation
If you find our works useful in your reasearch, please consider citing:  
```
@article{qiu2020peeking,
  title={Peeking into occluded joints: A novel framework for crowd pose estimation},
  author={Qiu, Lingteng and Zhang, Xuanye and Li, Yanran and Li, Guanbin and Wu, Xiaojun and Xiong, Zixiang and Han, Xiaoguang and Cui, Shuguang},
  journal={arXiv preprint arXiv:2003.10506},
  year={2020}
}
```

