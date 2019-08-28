from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import cv2

__all__ = ['part_segment']





def part_segment(fpath):

	"""
	crop the full densepose uv image to 24 parts
	:param fpath: the path of full densepose uv image
	:return: 24 parts
	"""
	
	path = cfg.data.densepose
    sources = cfg.data.sources
    target = osp.join(path,sources)

    dp_image = cv2.imread(fpath)


    parts = []

    for i in range(24):

    	row = i / 6
    	col = i % 6

    	part = dp_image[:,row*32:(row+1),col*32:(col+1)*32]


    return parts


