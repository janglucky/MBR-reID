from yacs.config import CfgNode as CN


def get_default_config():

    cfg = CN()


    #data
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']

    #loss
    cfg.loss = CN()
    cfg.loss.name = ['triplet', 'cross_entropy', 'l1']
    cfg.loss.weights = ['1','1','1']

    #optim
    cfg.train = CN()
    cfg.train.optim = 'sgd'
    cfg.train.warmup_mode = 'linear'

    return cfg