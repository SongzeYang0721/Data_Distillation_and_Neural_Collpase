import train_1st_order
import train_2nd_order

def train(args,model,trainloader):
    print("Initalize training...")
    if args.optimizer != 'LBFGS':
        print("Optimizer: ", args.optimizer)
        train_1st_order.train(args,model,trainloader)
    else:
        print("Optimizer: ", args.optimizer)
        train_2nd_order.train(args,model,trainloader)