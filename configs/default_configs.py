import argparse
from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()


    # gpu
    cfg.use_gpu = True

    # model
    cfg.model = CN()
    cfg.model.name = 'AENet'
    cfg.model.pretrained = False  # automatically load pretrained model weights if available
    cfg.model.load_weights = 'log/model.pth.tar-149'  # path to model weights
    cfg.model.resume = 'log/checkpoint.pth'  # path to checkpoint for resume training

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = '/home/jy/datasets'
    cfg.data.densepose = '/home/jy/datasets/DensePoseData'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 8  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_crop','random_flip','pad','random_erase']  # data augmentation
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False  # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False  # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomIdentitySampler'
    cfg.sampler.num_instances = 4  # number of instances per identity for RandomIdentitySampler

    # train
    cfg.train = CN()
    cfg.train.evaluate_epoch = 1
    cfg.train.optim = 'sgd'
    cfg.train.lr = 3e-2
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 320
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 128
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = ['classifier']  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ['classifier']  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [60, 130,300]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed
    cfg.train.warmup_epoch = 10 # model warmup
    cfg.train.warmup_begin_lr = 3e-4
    cfg.train.warmup_mode = 'linear'

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = True  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moments
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()

    ## criterions
    cfg.loss.name = ['softmax','softmax','softmax','triplet','triplet','triplet','triplet','triplet','triplet','sm' ]
    cfg.loss.weights = [1/3, 1/3,1/3,1/6,1/6,1/6,1/6,1/6,1/6,0.1*1/24]

    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    # cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    # cfg.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 32
    cfg.test.dist_metric = 'cosine'  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = True  # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = 10  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 1  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visactmap = False  # visualize CNN activation maps
    cfg.test.layer = [0,1,2]

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'densepose': cfg.data.densepose,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
    }

def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'warmup_epoch': cfg.train.warmup_epoch,
        'warmup_begin_lr': cfg.train.warmup_begin_lr,
        'warmup_mode': cfg.train.warmup_mode
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'visactmap': cfg.test.visactmap,
    }