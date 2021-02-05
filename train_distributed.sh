python -m torch.distributed.launch --nproc_per_node=2 \
    ./tools/train_pifu.py --dist --current 12 --resume --config ./configs/PIFu_Render_People_HG.py    