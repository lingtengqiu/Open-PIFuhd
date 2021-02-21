# Open-PIFu
[PIFu: Pixel-Aligned Implicit Function forHigh-Resolution Clothed Human Digitization](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_PIFu_Pixel-Aligned_Implicit_Function_for_High-Resolution_Clothed_Human_Digitization_ICCV_2019_paper.pdf)(CVPR2019) 
![](show_image/pipeline.png)





# PIFU results comparision

|                                     | IoU      | ACC     | recall  | P2S    | Normal | Chamfer |
| ----------------------------------- | -------- | ------- | ------- | ------ | :----: | :-----: |
| Origin                              | 0.737    | 0.875   | 0.820   | 1.812  | 0.1506 |  2.08   |
| only-gpu                            | 0.748    | 0.880   | 0.856   | 1.801  | 0.1446 |  2.00   |
| Coarse-PIFu(+Front and back normal) | 0.867498 | 0.93216 | 0.92492 | 1.2397 | 0.1201 | 1.4013  |

