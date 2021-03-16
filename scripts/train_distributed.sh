# python -m torch.distributed.launch --nproc_per_node=2 \
#     ./tools/train_pifu.py --dist --current 6 --resume --config ./configs/PIFu_Render_People_HG.py    
# python -m torch.distributed.launch --nproc_per_node=2 \
#     ./tools/train_pifu.py --dist --current 6 --resume --config ./configs/PIFu_Render_People_HG_adam_bce.py    