import os
import glob
import random

import scipy
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from tqdm import tqdm
from soundfile import SoundFile


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    # label_stack = []
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    # return data, torch.tensor(label_stack), seq_length
    return data, label, seq_length


def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, labels, seq_length

def collate_fn_atten_kd(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, short_seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    data_short = rnn_utils.pad_sequence(short_seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, data_short, labels, seq_length



def collate_fn_cnn_atten(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    label_cnn = rnn_utils.pad_sequence(label, batch_first=True, padding_value=255)
    labels = 0
    label_cnn_ = 0
    for i in range(len(label)):
        if i == 0:
            labels = label[i]
            label_cnn_ = label_cnn[0]
        else:
            labels = torch.cat((labels, label[i]),-1)
            label_cnn_ = torch.cat((label_cnn_, label_cnn[i]),-1)
    return data, labels, label_cnn_, seq_length


class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[1] for i in lines]
            self.seq_len_list = [i.split()[2] for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)


class KD_RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[1] for i in lines]
            self.seq_len_list = [i.split()[2] for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        short_feature = torch.from_numpy(np.load(feature_path, allow_pickle=True)[:5,:])
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, short_feature, label, seq_len

    def __len__(self):
        return len(self.label_list)



def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_student(seq_lens, batch_size, mask_type='fix', win_len=15):
  max_len = seq_lens[0]
  atten_mask = torch.ones([batch_size, max_len, max_len])
  if mask_type == 'fix':
    for i in range(batch_size):
      if seq_len>win_len:
        atten_mask[i, 0:win_len, 0:win_len] = 0
      else:
        atten_mask[i, :seq_len, :seq_len] = 0
  elif mask_type == 'random':
    for i in range(batch_size):
      seq_len = seq_lens[i]
      if seq_len>win_len:
          rest_len = seq_len - win_len
          start = random.randint(0, rest_len)
          end = start + win_len
          atten_mask[i, start:end, start:end] = 0
      else:
          atten_mask[i, :seq_len, :seq_len] = 0
  return atten_mask.bool()
