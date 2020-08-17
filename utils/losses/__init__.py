from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .sm_loss import SMLoss
from configs.default_configs import get_default_config

#criterion, outputs, targets, loss_weighs
def DeepSupervision(criterions, xs, ys, ws):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterions: all loss functions
        xs: tuple of inputs
        ys: ground truths
        ws: weight of losses
    """
    loss = 0.

    cfg = get_default_config()

    class_num = xs[0].size(1)

    for i in range(len(ys)):

        if criterions[i]=='softmax':
            criterion = CrossEntropyLoss(
                num_classes=class_num,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        elif criterions[i] == 'triplet':
            criterion = TripletLoss(margin=cfg.loss.triplet.margin)

        elif criterions[i] == 'sm':
            criterion = SMLoss()

        loss += ws[i]*criterion(xs[i], ys[i])

    return loss