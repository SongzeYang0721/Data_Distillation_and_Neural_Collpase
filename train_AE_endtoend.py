import sys
import gc
import wandb
from utils import *
import matplotlib.pyplot as plt
from Visualization import *

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

def AE_trainer_1st(args_encoder, args_decoder, autoencoder, trainloader, epoch_id, criterion_encoder, criterion_decoder, optimizer, scheduler):

    losses_encoder = AverageMeter()
    losses_AE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args_decoder.epochs, scheduler.get_last_lr()[-1]))

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args_decoder.device), targets.to(args_decoder.device)

        autoencoder.train()

        reconstruction, outputs = autoencoder(inputs)
        
        loss_AE = criterion_decoder(reconstruction, inputs)
        # loss = criterion(outputs, inputs).to(args.device)
        # loss = nn.functional.binary_cross_entropy(outputs,inputs,reduction="mean").to(args.device)
        if args_encoder.sep_decay:
            loss_encoder = loss_compute(args_encoder, autoencoder.encoder, criterion_encoder, outputs, targets)
        else:
            if args_encoder.loss == 'CrossEntropy':
                loss_encoder = criterion_encoder(outputs, targets)
            elif args_encoder.loss == 'MSE':
                loss_encoder = criterion_encoder(outputs, nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args_decoder.device))
        loss = loss_encoder + loss_AE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del reconstruction, outputs

        # measure accuracy and record loss
        autoencoder.eval()
        losses_AE.update(loss_AE.detach().item(), inputs.size(0))
        del loss_AE
        with torch.no_grad():
            outputs = autoencoder.encoder(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses_encoder.update(loss_encoder.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    print('[epoch: %d] (%d/%d) | Loss(encoder): %.4f | Loss(AE): %.4f | top1: %.4f | top5: %.4f ' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses_encoder.avg, losses_AE.avg, top1.avg, top5.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses(encoder).avg":losses_encoder.avg, 
            "losses(AE).avg": losses_AE.avg,
            "LR":scheduler.get_last_lr()[-1],
            "top1.avg":top1.avg,
            "top5.avg":top5.avg
        })

    scheduler.step()

    del losses_AE, losses_encoder, outputs

def AE_trainer_2nd(args_encoder, args_decoder, autoencoder, trainloader, epoch_id, criterion_encoder, criterion_decoder, optimizer):

    losses_encoder = AverageMeter()
    losses_AE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args_decoder.epochs))

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args_decoder.device), targets.to(args_decoder.device)

        autoencoder.train()

        def closure():
            reconstruction, outputs = autoencoder(inputs)
            
            loss = criterion_decoder(reconstruction, inputs) #+ weight_decay(args, autoencoder)
            # loss = criterion(outputs, inputs).to(args.device)
            # loss = nn.functional.binary_cross_entropy(outputs,inputs,reduction="mean").to(args.device)

            if args_encoder.loss == 'CrossEntropy':
                loss += criterion_encoder(outputs, targets) + weight_decay(args_encoder, autoencoder.encoder)
            elif args_encoder.loss == 'MSE':
                loss += criterion_encoder(outputs, nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args_encoder.device)) \
                       + weight_decay(args_encoder, autoencoder.encoder)
            
            optimizer.zero_grad()
            loss.backward()

            return loss
        
        optimizer.step(closure)

        # measure accuracy and record loss
        autoencoder.eval()
        with torch.no_grad():
            reconstruction, _ = autoencoder(inputs)
        loss_AE = criterion_decoder(reconstruction, inputs)
        losses_AE.update(loss_AE.detach().item(), inputs.size(0))
        del loss_AE
        with torch.no_grad():
            outputs = autoencoder.encoder(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))

        if args_encoder.loss == 'CrossEntropy':
            loss_encoder = criterion_encoder(outputs[0], targets) + weight_decay(args_encoder, autoencoder.encoder)
        elif args_encoder.loss == 'MSE':
            loss_encoder = criterion_encoder(outputs[0], nn.functional.one_hot(targets,num_classes=outputs[0].shape[1]).type(torch.FloatTensor).to(args_encoder.device)) \
                   + weight_decay(args_encoder, autoencoder.encoder)

        losses_encoder.update(loss_encoder.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        del loss, outputs
    
    print('[epoch: %d] (%d/%d) | Loss(encoder): %.4f | Loss(AE): %.4f | top1: %.4f | top5: %.4f ' %
          (epoch_id + 1, batch_idx + 1, len(trainloader), losses_encoder.avg, losses_AE.avg, top1.avg, top5.avg))

    if 'wandb' in sys.modules:
        wandb.log({
            "losses(encoder).avg":losses_encoder.avg, 
            "losses(AE).avg": losses_AE.avg,
            "top1.avg":top1.avg,
            "top5.avg":top5.avg
        })


def AE_train_endtoend(args_encoder,args_decoder,model,trainloader,visualize = False):

    criterion_encoder = make_criterion(args_encoder)
    criterion_decoder = make_criterion(args_decoder)
    optimizer = make_optimizer(args_decoder, model)
    scheduler = make_scheduler(args_decoder, optimizer)

    print('# of model parameters: ' + str(count_network_parameters(model)))
    print('--------------------- Training -------------------------------')

    if visualize:
        indices = random_sample_images_index(trainloader)

    for epoch_id in range(args_decoder.epochs):

        torch.cuda.empty_cache()
        gc.collect()

        if args_decoder.optimizer == 'LBFGS':
            AE_trainer_2nd(args_encoder, args_decoder, model, trainloader, epoch_id, criterion_encoder, criterion_decoder, optimizer)
        else:
            AE_trainer_1st(args_encoder, args_decoder, model, trainloader, epoch_id, criterion_encoder, criterion_decoder, optimizer, scheduler)

        # Visualization check
        if visualize:
            inputs, labels = images_from_index(trainloader.dataset, indices)
            inputs, labels = inputs.to(args_decoder.device), labels.to(args_decoder.device)
            with torch.no_grad():
                # Taking a subset for visualization
                reconstruction, _ = model(inputs)
                visualize_images(inputs.cpu(),labels.cpu(), False)
                visualize_images(reconstruction.cpu(),labels.cpu(), False)
                del reconstruction, inputs, labels
        torch.save(model.encoder.state_dict(), args_encoder.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")
        if epoch_id % 100 == 0:
            torch.save(model.decoder.state_dict(), args_decoder.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pt")
        print(f"Memory cached in GPU: {torch.cuda.memory_reserved()}")
