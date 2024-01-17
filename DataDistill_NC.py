import train_1st_order
import sys

import torch

import models
from models.res_adapt import ResNet18_adapt
from utils import *
from args import parse_train_args
from datasets import make_dataset

def generate_support_data():
    inds = np.random.choice(50000, 500)
    x_support, y_support = X_TRAIN[inds], Y_TRAIN[inds]
    data = 
    for _ in range(self.num_data_steps):
        distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                device=state.device, requires_grad=True)
        # print("distill_data.shape", distill_data.shape)
        self.data.append(distill_data)
        self.params.append(distill_data)

def argument_with_support():
    

def main():
    args = parse_train_args()

    set_seed(manualSeed = args.seed)

    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=args.SOTA)


    
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=num_classes).to(device)
    elif args.model == "ResNet18_adapt":
        model = ResNet18_adapt(width = args.width, num_classes=num_classes, fc_bias=args.bias).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    train(args, model, trainloader)


if __name__ == "__main__":
    main()
