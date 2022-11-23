import os
import logging
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup, get_nodecay_schedule, RandomSampler
from models.soc.soc import SoC
from lib.datasets.iNatDataset import iNatDataset
from torchvision import  transforms
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 

def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    # divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)
    
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 
        
        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    
    global best_acc1
    args.gpu = gpu
    
    # set random seed for reproducibility 
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    
    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank
        
        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    
    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard')
        logger_level = "INFO"
    
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    
    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net, 
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})
    
    model = SoC(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.ulb_loss_ratio,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')
        

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    if args.lr_decay=='cos':
        if args.pretrained:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    args.num_train_iter,
                                                    num_warmup_steps=args.num_train_iter*0.1)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    2**20,
                                                    num_warmup_steps=args.num_train_iter*0)
    else:
        scheduler = get_nodecay_schedule(optimizer)
    ## set SGD and cosine lr on SoC 
    model.set_optimizer(optimizer, scheduler)
    
    
    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)            
            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)
            
        else:
            model.train_model.feature_extractor = torch.nn.parallel.DistributedDataParallel(model.train_model.feature_extractor).cuda()
            model.eval_model.feature_extractor = torch.nn.parallel.DistributedDataParallel(model.eval_model.feature_extractor).cuda()
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)
              
    else:
        model.train_model.feature_extractor = torch.nn.DataParallel(model.train_model.feature_extractor).cuda()
        model.eval_model.feature_extractor = torch.nn.DataParallel(model.eval_model.feature_extractor).cuda()
    
    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")
    
    cudnn.benchmark = True


    # Construct Dataset & DataLoader
    loader_dict = {}
    if args.dataset=='semi_fungi' or args.dataset=='semi_aves':
        data_transforms = {
            'train': transforms.Compose([
    #             transforms.Resize(args.input_size), 
                transforms.RandomResizedCrop(args.input_size),
                # transforms.ColorJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(args.input_size), 
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        data_transforms['l_train'] = data_transforms['train']
        data_transforms['u_train'] = data_transforms['train']
        data_transforms['val'] = data_transforms['test']

        root_path = args.data_dir

        if args.trainval:
            ## following "A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification"
            ## l_train + val are used for labeled training data
            l_train = 'l_train_val'
        else:
            l_train = 'l_train'

        if args.unlabel == 'in':
            u_train = 'u_train_in'
        elif args.unlabel == 'inout':
            u_train = 'u_train'

        ## set val to test when using l_train + val for training
        if args.trainval:
            split_fname = ['test', 'test']
        else:
            split_fname = ['val', 'test']

        image_datasets = {split: iNatDataset(root_path, split_fname[i], args.dataset,
            transform=data_transforms[split], return_name=True) \
            for i,split in enumerate(['val', 'test'])}
        image_datasets['u_train'] = iNatDataset(root_path, u_train, args.dataset,
            transform=data_transforms['u_train'])
        image_datasets['l_train'] = iNatDataset(root_path, l_train, args.dataset,
            transform=data_transforms['train'], return_name=True)

        print("labeled data : {}, unlabeled data : {}".format(len(image_datasets['l_train']), len(image_datasets['u_train'])))
        print("validation data : {}, test data : {}".format(len(image_datasets['val']), len(image_datasets['test'])))

        num_classes = image_datasets['l_train'].get_num_classes() 
        
        print("#classes : {}".format(num_classes))

        loader_dict['train_lb'] = DataLoader(image_datasets['l_train'],
                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                        sampler=RandomSampler(len(image_datasets['l_train']), args.num_train_iter * args.batch_size))

        mu = args.uratio
        loader_dict['train_ulb'] = DataLoader(image_datasets['u_train'],
                        batch_size=args.batch_size * mu, num_workers=args.num_workers, drop_last=True,
                        sampler=RandomSampler(len(image_datasets['u_train']), args.num_train_iter * args.batch_size * mu))
        # loader_dict['val'] = DataLoader(image_datasets['val'],
                        # batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        loader_dict['eval_ulb'] = DataLoader(image_datasets['u_train'],
                        batch_size=args.batch_size * mu, shuffle=False, num_workers=args.num_workers, drop_last=False)
        loader_dict['eval'] = DataLoader(image_datasets['test'],
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False,
                        sampler=RandomSampler(len(image_datasets['test']), len(image_datasets['test'])))
    else:
        print('Not implemented')
    
    # set DataLoader on SoC
    model.set_data_loader(loader_dict)
    
    # If args.resume, load checkpoints 
    if args.resume:
        model.load_model(args.load_path, args.load_path_soc)
    
    # START TRAINING of SoC
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)
        
    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)
        
    logging.warning(f"GPU {args.rank} training is FINISHED")
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='')
    
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='soc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None, help='the path to checkpoints')
    parser.add_argument('--load_path_soc', type=str, default=None, help='the path to soc.pkl containing centroids and label_matrix (not necessary)')
    parser.add_argument('--overwrite', action='store_true')
    
    '''
    Training Configuration of SoC
    '''
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=400000, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10000,
                        help='evaluation frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=5,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--num_tracked_batch', type=int, default=5120, help='total number of batch tracked by CTT')
    parser.add_argument('--alpha', type=float, default=2.5, help='use {2.5,4} for {semi_aves,semi_fungi}')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    
    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=str, default='cos', help='use {cos,none} for {cosine decay,no decay}')
    parser.add_argument('--pretrained', action='store_true', default=False)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='resnet50', 
            help='use {resnet18/50/101,wrn,wrnvar,cnn13} for {ResNet-18/50/101,Wide ResNet,Wide ResNet-Var(WRN-37-2),CNN-13}')
    # for Wide ResNet
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    '''
    Data Configurations
    '''
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='semi_aves')
    # parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', default=224, type=int, 
            help='input image size')
    parser.add_argument('--unlabel', default='in', type=str, 
            choices=['in','inout'], help='U_in or U_in + U_out')
    parser.add_argument('--trainval', action='store_true', default=True,
            help='use {train+val,test,test} for {train,val,test}')
    
    '''
    multi-GPUs & Distrbitued Training
    '''
    
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
     
    args = parser.parse_args()
    main(args)
