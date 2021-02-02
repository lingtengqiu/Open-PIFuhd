import argparse
import torch.utils.data as data
parser = argparse.ArgumentParser(description='Parameters of List of PIFu')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--dataset', default='render_people', type=str,
                    help='Dataset choice: mpii | coco')
parser.add_argument('--debug', default=False, type=bool,
                    help='Print the debug information')

"----------------------------- Model options -----------------------------"

"----------------------------- Hyperparameter options -----------------------------"

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=2.5e-4, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')
parser.add_argument('--save_dirs', default='checkpoint', type=str,
                    help='where to save our project')
parser.add_argument('--load_dirs', type=str,
                    help='where to load our project')


"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=50, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainBatch', default=20, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=20, type=int,
                    help='Valid-batch size')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=512, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=512, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=128, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=128, type=int,
                    help='Output heatmap width')


"----------------------------- Distribution options -----------------------------"
parser.add_argument('--dist', action = 'store_true', help='distributed training or not')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--num_workers", default=64, type=int)

"----------------------------- checkpoint configs -----------------------------"
parser.add_argument('--resume', action = 'store_true',help="resume checkpoint")
parser.add_argument('--current', help='current training step',type=int,default=0)


"----------------------------- training visible setting -----------------------------"
parser.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
parser.add_argument('--freq_gallery', type=int, default=1, help='freqency of the error plot')

"----------------------------- PIFu configs -----------------------------"
parser.add_argument('--config', help='train config file path')
opt = parser.parse_args()

