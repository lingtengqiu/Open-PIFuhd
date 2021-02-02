CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 \
    ./tools/test_pifu.py --dist  --config ./configs/PIFu_Carton_HG.py    