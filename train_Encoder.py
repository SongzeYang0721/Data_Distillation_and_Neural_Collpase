import sys

import torch

import models
from models.res_adapt import ResNet18_adapt
from utils import *
from args import parse_train_args
from data.datasets import make_dataset

import wandb

def loss_compute(args, model, criterion, outputs, targets):
    if args.loss == 'CrossEntropy':
        loss = criterion(outputs[0], targets)
    elif args.loss == 'MSE':
        loss = criterion(outputs[0], nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args.device))

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        w = model.fc.weight
        b = model.fc.bias
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss += lamb * (torch.sum(w ** 2) + torch.sum(b ** 2)) + lamb_feature * torch.sum(features ** 2)

    return loss

def weight_decay(args, model):

    penalty = 0
    for p in model.parameters():
        if p.requires_grad:
            penalty += 0.5 * args.weight_decay * torch.norm(p) ** 2

    return penalty.to(args.device)

def trainer_1st(args, model, trainloader, epoch_id, criterion, optimizer, scheduler):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()
        outputs = model(inputs)
        
        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets)
        else:
            if args.loss == 'CrossEntropy':
                loss = criterion(outputs[0], targets)
            elif args.loss == 'MSE':
                loss = criterion(outputs[0], nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, int(args.num_classes/2)))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # if batch_idx % 10 == 0:
        #     print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
        #                    (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)
    
    print('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top%d: %.4f ' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, int(args.num_classes/2), top5.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "LR":scheduler.get_last_lr()[-1],
            "losses.avg":losses.avg, 
            "top1.avg":top1.avg,
            "top5.avg":top5.avg
        })
    
    
    scheduler.step()

def trainer_2nd(args, model, trainloader, epoch_id, criterion, optimizer):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args.epochs))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()

        def closure():
            outputs = model(inputs)

            if args.loss == 'CrossEntropy':
                loss = criterion(outputs[0], targets) + weight_decay(args, model)
            elif args.loss == 'MSE':
                loss = criterion(outputs[0], nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args.device)) \
                       + weight_decay(args, model)

            optimizer.zero_grad()
            loss.backward()

            return loss

        optimizer.step(closure)

        # measure accuracy and record loss
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, int(args.num_classes/2)))

        if args.loss == 'CrossEntropy':
            loss = criterion(outputs[0], targets) + weight_decay(args, model)
        elif args.loss == 'MSE':
            loss = criterion(outputs[0], nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args.device)) \
                   + weight_decay(args, model)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # if batch_idx % 10 == 0:
        #     print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
        #                    (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)
    
    print('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top%d: %.4f ' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, int(args.num_classes/2) ,top5.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses.avg":losses.avg, 
            "top1.avg":top1.avg,
            "top5.avg":top5.avg
        })

def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    print('# of model parameters: ' + str(count_network_parameters(model)))
    print('--------------------- Training -------------------------------')
    for epoch_id in range(args.epochs):
        if args.optimizer == 'LBFGS':
            trainer_2nd(args, model, trainloader, epoch_id, criterion, optimizer)
        else:
            trainer_1st(args, model, trainloader, epoch_id, criterion, optimizer, scheduler)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pt")


def main():
    args = parse_train_args()

    set_seed(manualSeed = args.seed)

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=args.SOTA)

    if args.optimizer == 'LBFGS':
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA).to(device)
    else:
        if args.model == "MLP":
            model = models.__dict__[args.model](hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=num_classes).to(device)
        elif args.model == "ResNet18_adapt":
            model = ResNet18_adapt(width = args.width, num_classes=num_classes, fc_bias=args.bias).to(device)
        else:
            model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA).to(device)
    
    train(args, model, trainloader)


if __name__ == "__main__":
    main()
