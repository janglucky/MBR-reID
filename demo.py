# from __future__ import absolute_import

from utils.data import ImageDataManager
from configs.default_configs import imagedata_kwargs,get_default_config


def build_datamanager(cfg):
    """
    construcure the dataloader
    :param cfg:
    :return:
    """
    return ImageDataManager(**imagedata_kwargs(cfg))

cfg = get_default_config()
datamanager = build_datamanager(cfg)