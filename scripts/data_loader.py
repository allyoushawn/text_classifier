#!/usr/bin/env python3
from collections import defaultdict
import random
import torch
from torch.utils import data







class Dataset(data.Dataset):

    def __init__(self, filename, init_w2i, init_t2i):
        self.w2i = init_w2i
        self.t2i = init_t2i
        self.data =  list(self.read_dataset(filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words = self.data[index][0]
        label = self.data[index][1]
        word_tensor = torch.tensor(words, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        #return self.data[index][0], self.data[index][1]
        return word_tensor, label_tensor

    def read_dataset(self, filename):
      with open(filename, "r",  encoding="utf-8") as f:
        for line in f:
          tag, words = line.lower().strip().split(" ||| ")
          # yields ([w1_idx, w2_idx, ..., wn_idx], tag)
          yield ([self.w2i[x] for x in words.split(" ")], self.t2i[tag])


def get_key(item):
    return len(item[0])

def my_collate(batch):
    batch = sorted(batch, key=get_key, reverse=True)
    '''
    data = []
    for item in batch:
        words = []
        for word in item[0]:
            if random.random() < 0.1:
                words.append(1)
            else:
                words.append(word)
        data.append(torch.tensor(words))
    '''
    data = [item[0] for item in batch]

    lengths = [torch.tensor(len(item[0]), dtype=torch.long) for item in batch]
    lengths = torch.stack(lengths)
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    target = [item[1] for item in batch]
    target = torch.stack(target)
    return [data, target, lengths]
