srun -J srun -p p-V100 -N 1 --cpus-per-task=4 --gres=gpu:1 --pty \
                python train.py --name image2backnormal1024 --no_flip --no_front --dataroot ../Garment/render_gen_1024_train/ --label_nc 0


