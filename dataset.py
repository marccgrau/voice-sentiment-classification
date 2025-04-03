# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:59:33 2023

@author: 28257
"""

import pickle
from typing import Tuple

import torch
from torch.utils.data import Dataset


class WhisPArDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, "rb+") as fp:
            mel_list, label_list = pickle.load(fp)

        self.mel_list = mel_list
        self.label_list = [
            torch.tensor(label, dtype=torch.long) for label in label_list
        ]

    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        mel = self.mel_list[index]
        label = self.label_list[index]
        return mel, label
