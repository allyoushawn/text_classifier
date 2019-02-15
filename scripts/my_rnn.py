#!/usr/bin/env python3
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time
import matplotlib.pyplot as plt
from data_loader import Dataset, my_collate
from torch.utils import data
from locked_dropout import LockedDropout
import pickle
import os

import pdb






def save_model(model, model_save_path):
    model.save(model_save_path)
    state = {
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state,
               os.path.join(model_save_path, 'opt.pth'))


def load_model(model, model_save_path, device):
    model = model.load(model_save_path, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr)
    print('restore parameters of the optimizers', file=sys.stderr)
    # You may also need to load the state of the optimizer saved before
    state = torch.load(
        os.path.join(model_save_path, 'opt.pth'))
    optimizer.load_state_dict(state['optimizer'])


if __name__ == '__main__':
    # Functions to read in the corpus
    # The defaultdict would return len(w2i) when there is a
    # new element comes in
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    PADDING = w2i["<PADDING>"]
    UNK = w2i["<unk>"]


    # Read in the data
    train_dataset = Dataset("../topicclass/train.txt", w2i, t2i)
    w2i = train_dataset.w2i
    t2i = train_dataset.t2i
    n_words = len(w2i)
    n_tags = len(t2i)

    # The following code is initializing a dict with w2i.
    # Then, if there is a new words, return UNK
    w2i = defaultdict(lambda: UNK, w2i)
    dev_dataset = Dataset("../topicclass/dev.txt", w2i, t2i)
    n_embeds = 50
    n_hid = 64
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6,
              'collate_fn': my_collate}
    train_generator = data.DataLoader(train_dataset, **params)
    dev_generator = data.DataLoader(dev_dataset, **params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('embedding.pkl', 'rb') as fp:
        embed_matrix = pickle.load(fp)
    model = SimpleRNN(n_words, n_tags, n_embeds, n_hid, embed_matrix).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    tr_tags_freq = [0] * n_tags


    step = 0
    for epoch in range(100):
        start = time.time()
        model.train()
        for local_batch, tag, lengths in train_generator:
            train_loss = 0.0
            train_correct = 0
            local_batch = local_batch.cuda()
            tag = tag.cuda()
            optimizer.zero_grad()
            output = model(local_batch, lengths)
            for i in range(len(local_batch)):
                predict = output[i].argmax().item()
                if predict == tag[i].cpu().item():
                    train_correct += 1
            loss = criterion(output, tag)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step += 1
            train_loss += loss.item()
            if step % 1000 == 0:
                print("Step %r: train loss/sent=%.4f, tr. acc=%.4f"
                  % (step, train_loss/len(tag), train_correct / len(tag)))
        model.eval()
        #print('Avg. grad: {:.4f}'.format(sum_grad / sum_param))
        # Perform testing
        hit = 0.0
        dev_num = 0
        hit = 0.0
        dev_num = 0
        for local_batch, tag, lengths in dev_generator:
            local_batch = local_batch.cuda()
            tag = tag.cuda()
            output = model(local_batch, lengths)
            for i in range(len(local_batch)):
                dev_num += 1
                predict = output[i].argmax().item()
                if predict == tag[i].cpu().item():
                    hit += 1
        print("Finish epoch {}, time={:.2f}, dev. acc={:.4f}".format(epoch, time.time() - start, hit/dev_num))
