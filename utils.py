import os
from glob import glob
from shutil import copyfile

import bcolz
import torch
import numpy as np


def make_validation(path, num_valid):
    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')
    if not os.path.exists(valid_path):
        os.mkdir(valid_path)

    class_folders = glob(os.path.join(train_path, '*'))
    for c in class_folders:
        os.mkdir(os.path.join(valid_path, c.split(os.sep)[-1]))

    images = glob(os.path.join(train_path, '*', '*.jpg'))
    shuffle = np.random.permutation(images)
    for i in range(num_valid):
        os.rename(shuffle[i], os.path.join(valid_path, *shuffle[i].split(os.sep)[-2:]))


def make_sample(path, train_size, valid_size):
    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')
    sample_path = os.path.join(path, 'sample')
    sample_path_train = os.path.join(sample_path, 'train')
    sample_path_valid = os.path.join(sample_path, 'valid')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    if not os.path.exists(sample_path_train):
        os.mkdir(sample_path_train)
    if not os.path.exists(sample_path_valid):
        os.mkdir(sample_path_valid)

    class_folders = glob(os.path.join(train_path, '*'))
    for c in class_folders:
        os.mkdir(os.path.join(sample_path_train, c.split(os.sep)[-1]))
        os.mkdir(os.path.join(sample_path_valid, c.split(os.sep)[-1]))

    train_images = glob(os.path.join(train_path, '*', '*.jpg'))
    valid_images = glob(os.path.join(valid_path, '*', '*.jpg'))
    shuffled_train = np.random.permutation(train_images)
    shuffled_valid = np.random.permutation(valid_images)
    for i in range(train_size):
        copyfile(shuffled_train[i], os.path.join(
            sample_path_train, *shuffled_train[i].split(os.sep)[-2:]))

    for i in range(valid_size):
        copyfile(shuffled_valid[i], os.path.join(
            sample_path_valid, *shuffled_valid[i].split(os.sep)[-2:]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1 - mx) / 1, mx)
    return clipped / clipped.sum(axis=1)[:, np.newaxis]


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]
