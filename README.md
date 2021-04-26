# Pix2PixHD for RenderPeople
It is used for Generating the back normal and front normal of Render People


## Prerequisites
- Linux or macOS
- Python 2 or 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```

### Testing
```bash
# test back normal
python test.py --name image2backnormal1024 --no_instance --no_front [--which_epoch 20] --dataroot ../Garment/render_gen_1024_test/ --label_nc 0 --how_many 100000

# test front normal
python test.py --name image2normal1024 --no_instance --dataroot ../Garment/render_gen_1024_test/ --label_nc 0 --which_epoch 1 --how_many 100000
```


### Dataset
- Render people datasets(more details see [Open-PIFuhd](https://github.com/lingtengqiu/Open-PIFuhd))


### Training
- Train a model at 1024 x 1024 resolution (render people):
```bash
# For front normal
python train.py --name image2normal1024 --no_flip  --dataroot ../Garment/render_gen_1024_train/ --label_nc 0

# For back normal
python train.py --name image2backnormal1024 --no_flip --no_front --dataroot ../Garment/render_gen_1024_train/ --label_nc 0
```
We fine that only employ  Vgg loss and L1 loss are better that GAN loss for back normal task



## Pretrain

- [Front]()
- [Back]()

### Visualization

|                            input                             |                             real                             |                             fake                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![epoch018_input_label](./checkpoints/image2backnormal1024/web/images/epoch018_input_label.jpg) | ![epoch018_real_image](./checkpoints/image2backnormal1024/web/images/epoch018_real_image.jpg) | ![epoch018_synthesized_image](./checkpoints/image2backnormal1024/web/images/epoch018_synthesized_image.jpg) |

  


## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
