import sys
import gc
import wandb
from utils import *
import matplotlib.pyplot as plt
from Visualization import *

def weight_decay(args, model):

    penalty = 0
    for p in model.parameters():
        if p.requires_grad:
            penalty += 0.5 * args.weight_decay * torch.norm(p) ** 2

    return penalty.to(args.device)

def AE_trainer_1st(args, autoencoder, trainloader, epoch_id, criterion, optimizer, scheduler):

    losses = AverageMeter()

    print('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]))

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        autoencoder.train()

        outputs = autoencoder(inputs)
        
        loss = criterion(outputs, inputs)
        # loss = criterion(outputs, inputs).to(args.device)
        # loss = nn.functional.binary_cross_entropy(outputs,inputs,reduction="mean").to(args.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        autoencoder.eval()
        losses.update(loss.detach().item(), inputs.size(0))
        del loss, outputs
    
    print('[epoch: %d] (%d/%d) | Loss: %.4f |' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses.avg":losses.avg, 
            "LR":scheduler.get_last_lr()[-1]
        })

    scheduler.step()

def AE_trainer_2nd(args, autoencoder, trainloader, epoch_id, criterion, optimizer):

    losses = AverageMeter()

    print('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args.epochs))

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        autoencoder.train()

        def closure():
            outputs = autoencoder(inputs)
            
            loss = criterion(outputs, inputs) #+ weight_decay(args, autoencoder)
            # loss = criterion(outputs, inputs).to(args.device)
            # loss = nn.functional.binary_cross_entropy(outputs,inputs,reduction="mean").to(args.device)

            optimizer.zero_grad()
            loss.backward()

            return loss
        
        optimizer.step(closure)

        # measure accuracy and record loss
        autoencoder.eval()
        outputs = autoencoder(inputs)
        with torch.no_grad():
            loss = criterion(outputs, inputs)
        losses.update(loss.detach().item(), inputs.size(0))
        del loss, outputs
    
    print('[epoch: %d] (%d/%d) | Loss: %.4f |' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses.avg":losses.avg
        })


def AE_train(args, model, trainloader, visualize = False):
    
    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    print('# of model parameters: ' + str(count_network_parameters(model)))
    print('--------------------- Training -------------------------------')

    if visualize:
        indices = random_sample_images_index(trainloader)

    for epoch_id in range(args.epochs):

        torch.cuda.empty_cache()
        gc.collect()

        if args.optimizer == 'LBFGS':
            AE_trainer_2nd(args, model, trainloader, epoch_id, criterion, optimizer)
        else:
            AE_trainer_1st(args, model, trainloader, epoch_id, criterion, optimizer, scheduler)

        # Visualization check
        if visualize:
            inputs, labels = images_from_index(trainloader.dataset, indices)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            with torch.no_grad():
                # Taking a subset for visualization
                outputs = model(inputs)
                visualize_images(inputs.cpu(),labels.cpu(), False)
                visualize_images(outputs.cpu(),labels.cpu(), False)
                del outputs, inputs, labels

        if epoch_id % 40 == 0:
            torch.save(model.decoder.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pt")
        print(f"Memory cached in GPU: {torch.cuda.memory_reserved()}")

