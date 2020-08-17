from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import torch


AVAI_SCH = ['single_step', 'multi_step']


# def build_lr_scheduler(optimizer, lr_scheduler, stepsize, gamma=0.1):
#     """A function wrapper for building a learning rate scheduler.
#
#     Args:
#         optimizer (Optimizer): an Optimizer.
#         lr_scheduler (str): learning rate scheduler method. Currently supports
#             "single_step" and "multi_step".
#         stepsize (int or list): step size to decay learning rate. When ``lr_scheduler`` is
#             "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
#             "multi_step", ``stepsize`` is a list.
#         gamma (float, optional): decay rate. Default is 0.1.
#
#     Examples::
#         >>> # Decay learning rate by every 20 epochs.
#         >>> scheduler = torchreid.optim.build_lr_scheduler(
#         >>>     optimizer, lr_scheduler='single_step', stepsize=20
#         >>> )
#         >>> # Decay learning rate at 30, 50 and 55 epochs.
#         >>> scheduler = torchreid.optim.build_lr_scheduler(
#         >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
#         >>> )
#     """
#     if lr_scheduler not in AVAI_SCH:
#         raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))
#
#     if lr_scheduler == 'single_step':
#         if isinstance(stepsize, list):
#             stepsize = stepsize[-1]
#
#         if not isinstance(stepsize, int):
#             raise TypeError(
#                 'For single_step lr_scheduler, stepsize must '
#                 'be an integer, but got {}'.format(type(stepsize))
#             )
#
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=stepsize, gamma=gamma
#         )
#
#     elif lr_scheduler == 'multi_step':
#         if not isinstance(stepsize, list):
#             raise TypeError(
#                 'For multi_step lr_scheduler, stepsize must '
#                 'be a list, but got {}'.format(type(stepsize))
#             )
#
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer, milestones=stepsize, gamma=gamma
#         )
#
#     elif lr_scheduler == 'man_step':
#         if not isinstance(stepsize,list):
#             raise TypeError(
#                 'For multi_step lr_scheduler, stepsize must '
#                 'be a list, but got {}'.format(type(stepsize))
#             )
#         scheduler = LRScheduler
#
#     return scheduler



class build_lr_scheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    warmup_epoch: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: string
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """

    def __init__(self, optimizer, stepsize=(30, 60), gamma=0.1,
                 warmup_epoch=0, warmup_begin_lr=3e-4, warmup_mode='linear'):

        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']

        self.learning_rate = optimizer.param_groups[0]['lr']
        self.stepsize = stepsize
        self.gamma = gamma
        assert isinstance(warmup_epoch, int)
        self.warmup_epoch = warmup_epoch

        self.warmup_final_lr = optimizer.param_groups[0]['lr']
        self.warmup_begin_lr = warmup_begin_lr
        if self.warmup_begin_lr > self.warmup_final_lr:
            raise ValueError("Base lr has to be higher than warmup_begin_lr")
        if self.warmup_epoch < 0:
            raise ValueError("Warmup steps has to be positive or 0")
        if warmup_mode not in ['linear', 'constant']:
            raise ValueError("Supports only linear and constant modes of warmup")
        self.warmup_mode = warmup_mode

    def adjust_lr(self, num_epoch):

        # model warmup
        if self.warmup_epoch > num_epoch:
            # warmup strategy
            if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_begin_lr + (self.warmup_final_lr - self.warmup_begin_lr) * \
                                     num_epoch / self.warmup_epoch
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_begin_lr
        # start
        else:
            count = sum([1 for s in self.stepsize if s <= num_epoch])
            self.learning_rate = self.base_lr * pow(self.gamma, count)

        self.optimizer.param_groups[0]['lr'] = self.learning_rate

        return self.learning_rate

    def learning_rate(self):

        return self.learning_rate
