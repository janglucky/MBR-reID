#-*- coding:UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import numpy as np
import tarfile
import zipfile
import copy
import torch

from utils.assist import read_image, mkdir_if_missing, download_url


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = [] # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            train.append((img_path, pid, camid))

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################
        return ImageDataset(
            train, self.query, self.gallery,
            transform=self.transform,
            mode=self.mode,
            combineall=False,
            verbose=False)


    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set() # 无序不重复元素集
        cams = set()
        for _, pid, camid, _ in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for _, pid, _ in self.gallery:
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError('{} dataset needs to be manually '
                               'prepared, please follow the '
                               'document to prepare this dataset'.format(self.__class__.__name__))

        mkdir_if_missing(dataset_dir)
        
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print('Downloading {} dataset to "{}"'.format(self.__class__.__name__, dataset_dir))
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        extension = osp.basename(fpath).split('.')[-1]



        try:
            if extension == 'tar':
                tar = tarfile.open(fpath)
                tar.extractall(path=dataset_dir)
                tar.close()
            elif extension == 'zip':
                zip_ref = zipfile.ZipFile(fpath, 'r')
                zip_ref.extractall(dataset_dir)
                zip_ref.close()
        except:
            raise Exception('ReadError:Expect file end with .tar or .zip,but found file {} end with {}'.format(fpath,extension))
            

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
              num_train_pids, len(self.train), num_train_cams,
              num_query_pids, len(self.query), num_query_cams,
              num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        
        img_path, pid, camid, densepose_path = self.data[index]
        img = read_image(img_path)
        densepose = None

        if self.mode == 'train':
            dp = read_image(densepose_path)

            densepose = []
            for i in range(24):
                row = int(i / 6)
                col = i % 6

                densepose.append(self.transform[1](dp.crop((col * 32, row * 32, (col + 1) * 32, (row + 1) * 32))))

        if self.transform is not None:
            img = self.transform[0](img)

        if not isinstance(densepose,list):
            # densepose = torch.zeros(img.shape)
            densepose = []
        return img, pid, camid, img_path, densepose

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        print('  ----------------------------------------')