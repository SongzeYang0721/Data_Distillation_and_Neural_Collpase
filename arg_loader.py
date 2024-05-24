import os
import shutil
import datetime
import torch
import numpy as np
import socket

class train_args:
    def __init__(self, model='resnet18', bias=True, ETF_fc=False, fixdim=0, SOTA=False,
                 width=1024, depth=6, gpu_id=0, seed=6, use_cudnn=True,
                 dataset='mnist', data_dir='~/data', uid=None, force=False,
                 epochs=200, batch_size=1024, loss='CrossEntropy', sample_size=None,
                 lr=0.1, patience=40, decay_type='step', gamma = 0.1, optimizer='SGD',
                 weight_decay=5e-4, sep_decay=False, feature_decay_rate=1e-4,
                 history_size=10, ghost_batch=128, device = "cpu",
                 
                 # additional data setting
                 num_classes = None,
                 classes_to_include = None,
                ):
        
        # parse train arguments
        self.model = model
        self.bias = bias
        self.ETF_fc = ETF_fc # fixed the last layer
        self.fixdim = fixdim # 
        self.SOTA = SOTA # store_true for data augmentation 

        # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
        self.width = width
        self.depth = depth

        # Hardware Setting
        self.gpu_id = gpu_id
        self.seed = seed
        self.use_cudnn = use_cudnn

        # Directory Setting
        self.dataset = dataset
        self.num_classes = num_classes
        self.classes_to_include = classes_to_include
        self.data_dir = data_dir
        self.uid = uid
        self.force = force # force to override the given uid

        # Learning Options
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.sample_size = sample_size

        # Optimization specifications
        self.lr = lr
        self.patience = patience # learning rate decay per N epochs
        self.decay_type = decay_type # learning rate decay type
        self.gamma = gamma # learning rate decay factor for step decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        
        # The following two should be specified when testing adding wd on Features
        self.sep_decay = sep_decay # whether to separate weight decay to last feature and last weights
        self.feature_decay_rate = feature_decay_rate # weight decay for last layer feature
        self.history_size = history_size # history size for LBFGS
        self.ghost_batch = ghost_batch # ghost size for LBFGS variants
        self.device = device
                
        if self.uid is None:
            unique_id = str(np.random.randint(0, 100000))
            print("revise the unique id to a random number " + str(unique_id))
            self.uid = unique_id
            timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
            # create the save path for the dijkstra
            if socket.gethostname() == "dijkstra":
                save_path = '/data5/model_weights/' + self.uid + '-' + timestamp
            else:
                save_path = './model_weights/' + self.uid + '-' + timestamp
        else:
            # create the save path for the dijkstra
            if socket.gethostname() == "dijkstra":
                save_path = '/data5/model_weights/' + str(self.uid)
            else:
                save_path = './model_weights/' + str(self.uid)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            if not self.force:
                raise ("please use another uid ")
            else:
                print("override this uid" + self.uid)
                for m in range(1, 10):
#                     print(os.path.exists(save_path + "/log.txt.bk" + str(m)))
                    if not os.path.exists(save_path + "/log.txt.bk" + str(m)):
                        shutil.copy(save_path + "/log.txt", save_path + "/log.txt.bk" + str(m))
                        shutil.copy(save_path + "/args.txt", save_path + "/args.txt.bk" + str(m))
                        break
        
        self.save_path = save_path
        self.log = save_path + "/log.txt"
        self.arg = save_path + "/args.txt"

        with open(self.log, 'w') as f:
            f.close()
        with open(self.arg, 'w') as f:
            print(self)
            print(self, file=f)
            f.close()
        if self.use_cudnn:
            print("cudnn is used")
            torch.backends.cudnn.benchmark = True
        else:
            print("cudnn is not used")
            torch.backends.cudnn.benchmark = False


class eval_args:
    def __init__(self, model='resnet18', bias=True, ETF_fc=False, fixdim=0, SOTA=False,
                 width=1024, depth=6, 
                 gpu_id=0, 
                 dataset='mnist', data_dir='~/data', load_path = None,
                 epochs=200, batch_size=1024, sample_size=None,
                ):
        
        # parameters
        # Model Selection
        self.model = model
        self.bias = bias
        self.ETF_fc = ETF_fc
        self.fixdim = fixdim
        self.SOTA = SOTA # store_true
        
        # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
        self.width = width
        self.depth = depth

        # Hardware Setting
        self.gpu_id = gpu_id

        # Directory Setting
        self.dataset = dataset
        self.data_dir = data_dir
        self.load_path = load_path

        # Learning Options
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_size = sample_size