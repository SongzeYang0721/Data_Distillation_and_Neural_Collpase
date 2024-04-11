import sys
import wandb
from utils import *

import numpy as np

def AE_trainer(args, autoencoder, trainloader, epoch_id, criterion, optimizer, scheduler=None):

    losses = AverageMeter()
    train_loss = []

    if args.optimizer == 'LBFGS':
        print('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args.epochs))
    else:
        print('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        autoencoder.train()
        outputs = autoencoder(inputs)
        
        loss = criterion(outputs, inputs).to(args.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        autoencoder.eval()
        losses.update(loss.item(), inputs.size(0))
        train_loss.append(loss.detach().cpu().numpy())
        

    print('[epoch: %d] (%d/%d) | Loss: %.4f | Train_loss: %.4f |' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, np.mean(train_loss)))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses.avg":losses.avg, 
            "LR":scheduler.get_last_lr()[-1]
        })

    if args.optimizer != 'LBFGS':
        scheduler.step()

def AE_train(args, model, trainloader):
    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    print('# of model parameters: ' + str(count_network_parameters(model)))
    print('--------------------- Training -------------------------------')
    for epoch_id in range(args.epochs):

        AE_trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler)
        torch.save(model.decoder.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")
