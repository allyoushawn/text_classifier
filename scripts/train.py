#!/usr/bin/env python3
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import time
import matplotlib.pyplot as plt
from data_loader import Dataset, my_collate
from torch.utils import data
import pickle
import sys
from nn_model import CNNModel, LSTMModel
import os
import numpy as np
import subprocess

import pdb





def save_model(model, model_save_path):
    model.save(model_save_path)
    state = {
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state,
               os.path.join(model_save_path, 'opt.pth'))


def load_model(model, model_save_path, init_lr, device):
    model = model.load(model_save_path, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr)
    print('restore parameters of the optimizers', file=sys.stderr)
    # You may also need to load the state of the optimizer saved before
    state = torch.load(
        os.path.join(model_save_path, 'opt.pth'))
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Please input model dir')
        quit()
    model_dir = sys.argv[1]
    try:
        subprocess.call('cp scripts/train.py scripts/nn_model.py {}'.format(
           model_dir), shell=True)
    except:
        print('Error in finding the dir: ' + model_dir)
        quit()

    # Functions to read in the corpus
    # The defaultdict would return len(w2i) when there is a
    # new element comes in

    # Alternative approach: add a dummpy string <PAD> into dictionary
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    PADDING = w2i["<PADDING>"]
    UNK = w2i["<unk>"]


    # Read in the data
    #train_dataset = Dataset("remove_stop_word/train.txt", w2i, t2i)
    train_dataset = Dataset("../topicclass/train.txt", w2i, t2i)
    w2i = train_dataset.w2i
    t2i = train_dataset.t2i
    n_words = len(w2i)
    n_tags = len(t2i)

    # The following code is initializing a dict with w2i.
    # Then, if there is a new words, return UNK
    w2i = defaultdict(lambda: UNK, w2i)
    #dev_dataset = Dataset("remove_stop_word/dev.txt", w2i, t2i)
    dev_dataset = Dataset("../topicclass/dev.txt", w2i, t2i)
    n_embeds = 300
    n_hid = 64
    lr_decay = 0.5
    init_lr = 1e-3
    params = {'batch_size': 512,
              'shuffle': True,
              'num_workers': 6,
              'collate_fn': my_collate}
    train_generator = data.DataLoader(train_dataset, **params)
    params = {'batch_size': 512,
              'shuffle': False,
              'num_workers': 6,
              'collate_fn': my_collate}
    dev_generator = data.DataLoader(dev_dataset, **params)

    with open('fast_text/embedding.pkl', 'rb') as fp:
        embed_matrix = pickle.load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel(n_words, n_tags, n_embeds, n_hid, embed_matrix).cuda()
    #model = LSTMModel(n_words, n_tags, n_embeds, n_hid, embed_matrix).cuda()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()
    tr_tags_freq = [0] * n_tags

    step = 0
    best_dev = -1
    retry = 0
    stop_train = False
    interval = 100
    for epoch in range(100):
        start = time.time()
        model.train()
        for local_batch, tag, lengths in train_generator:
            train_correct = 0
            local_batch = local_batch.cuda()
            tag = tag.cuda()
            optimizer.zero_grad()
            output = model(local_batch, lengths)


            loss = criterion(output, tag)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step += 1
            if step % interval == 0:
                model.eval()
                train_correct = 0
                output = model(local_batch, lengths)

                for i in range(len(local_batch)):
                    predict = output[i].argmax().item()
                    if predict == tag[i].cpu().item():
                        train_correct += 1

                loss = criterion(output, tag)
                print("Step %r: train loss/sent=%.4f, acc.=%.4f" 
                  % (step, loss.item(), train_correct / len(tag)))
                model.train()
            if step % (2 * interval) == 0:
                model.eval()
                # Perform testing
                hit = 0.0
                dev_num = 0
                error_freq = [0] * n_tags
                dev_loss = 0.
                for local_batch, tag, lengths in dev_generator:
                    local_batch = local_batch.cuda()
                    tag = tag.cuda()
                    output = model(local_batch, lengths)
                    loss = criterion(output, tag)
                    dev_loss += loss.item() * len(tag)
                    for i in range(len(local_batch)):
                        dev_num += 1
                        predict = output[i].argmax().item()
                        if predict == tag[i].cpu().item():
                            hit += 1
                        else: error_freq[tag[i]] += 1
                dev_acc = hit / dev_num
                print("Dev eval on step {}, loss/sent={:.4f},  acc.={:.4f}".format(step, dev_loss / dev_num, dev_acc))
                # Store model if it is getting better
                if dev_acc > best_dev:
                    save_model(model, model_dir)
                    retry = 0
                    best_dev = dev_acc
                else:
                    print('Model performance not getting better...')
                    retry += 1
                    print('Restoring to the previous state and halve the learning rate.')
                    init_lr *= lr_decay
                    model, optimizer = load_model(model, model_dir, init_lr, device)
                    if retry == 5:
                        print('Have tried too many times, stop training.')
                        stop_train = True
                        break

                model.train()
        if stop_train == True: break
        print("Finish epoch {}, time={:.2f}".format(epoch+1, time.time() - start))
        '''
        print('Error Dist.:')
        for i in range(n_tags):
            print('    Tag {}: {:.4f}'.format(i, error_freq[i] * 100 / sum(error_freq)))
        '''
    model.eval()
    print('Generating output files with best model.')
    init_flag = True
    scores_collection = None
    dev_ans = []
    hit = 0
    dev_num = 0
    dev_loss = 0.
    for local_batch, tag, lengths in dev_generator:
        local_batch = local_batch.cuda()
        tag = tag.cuda()
        output = model(local_batch, lengths)
        loss = criterion(output, tag)
        scores = F.softmax(output, dim=1).cpu().detach().numpy()
        if init_flag == True:
            scores_collection = scores
            init_flag = False
        else:
            scores_collection = np.concatenate([scores_collection, scores], axis=0)
        dev_loss += loss.item() * len(tag)
        for i in range(len(local_batch)):
            dev_num += 1
            predict = output[i].argmax().item()
            if predict == tag[i].cpu().item():
                hit += 1
            else: error_freq[tag[i]] += 1
            dev_ans.append(tag[i].cpu().item())
    dev_acc = hit / dev_num
    print("Dev eval on step {}, loss/sent={:.4f},  acc.={:.4f}".format(step, dev_loss / dev_num, dev_acc))
    print('Save the predict scores into np array as {}/dev_scores.pkl'.format(model_dir))
    with open(model_dir + '/dev_scores.pkl', 'wb') as fp:
        pickle.dump(scores_collection, fp)

    hit = 0
    init_flag = True
    scores_collection = None
    with open(model_dir + '/dev_ans.pkl', 'wb') as fp:
        pickle.dump(dev_ans, fp)
    try:
        for target_dir in ['cnn_model', 'rnn_model']:
            with open(target_dir + '/dev_scores.pkl', 'rb') as fp:
                if init_flag == True:
                    scores_collection = pickle.load(fp)
                    init_flag = False
                else:
                    scores_collection += pickle.load(fp)
        predicts = np.argmax(scores_collection, axis=1).tolist()
        for idx, predict in enumerate(predicts):
            if dev_ans[idx] == predict:
                hit += 1
        print('Ensemble model dev. acc: {:.4f}'.format(hit/len(predicts)))

    except:
        print('Could not perform ensemble yet')


