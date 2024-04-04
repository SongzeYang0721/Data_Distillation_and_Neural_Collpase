import sys
import wandb
from utils import *


def train_AE(args, autoencoder, trainloader, epoch_id, criterion, optimizer, scheduler):

    losses = AverageMeter()

#     wandb.log({
#         "epoch id":epoch_id + 1, 
#         "epochs":args.epochs,
#         "LR":scheduler.get_last_lr()[-1]
#     })

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
        outputs = autoencoder(inputs)
        losses.update(loss.item(), inputs.size(0))

    print('[epoch: %d] (%d/%d) | Loss: %.4f |' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses.avg":losses.avg, 
        })
    
    
    scheduler.step()
