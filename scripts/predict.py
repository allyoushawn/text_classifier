#!/usr/bin/env python3
import sys
import torch.optim as optim
from collections import defaultdict
import torch
from data_loader import Dataset, my_collate
from torch.utils import data
from nn_model import CNNModel, LSTMModel
import pickle
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    n_embeds = 50
    n_hid = 64
    init_lr = 1e-3

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

    #test_dataset = Dataset("remove_stop_word/test.txt", w2i, t2i)
    test_dataset = Dataset("../topicclass/test.txt", w2i, t2i)

    params = {'batch_size': 512,
              'shuffle': False,
              'num_workers': 6,
              'collate_fn': my_collate}
    dev_generator = data.DataLoader(dev_dataset, **params)
    test_generator = data.DataLoader(test_dataset, **params)
    with open('embedding.pkl', 'rb') as fp:
        embed_matrix = pickle.load(fp)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = CNNModel(n_words, n_tags, n_embeds, n_hid, embed_matrix).cuda()
    model = LSTMModel(n_words, n_tags, n_embeds, n_hid, embed_matrix).cuda()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    model, optimizer = load_model(model, model_dir, init_lr, device)
    criterion = nn.CrossEntropyLoss()
    hit = 0
    dev_num = 0
    init_flag = True
    scores_collection = None
    dev_ans = []
    hit = 0
    dev_num = 0
    dev_loss = 0
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
            dev_ans.append(tag[i].cpu().item())
    dev_acc = hit / dev_num
    print("Dev eval, loss/sent={:.4f},  acc.={:.4f}".format(dev_loss / dev_num, dev_acc))
    print('Save the predict scores into np array as {}/dev_scores.pkl'.format(model_dir))
    with open(model_dir + '/dev_scores.pkl', 'wb') as fp:
        pickle.dump(scores_collection, fp)

    hit = 0
    init_flag = True
    scores_collection = None
    with open(model_dir + '/dev_ans.pkl', 'wb') as fp:
        pickle.dump(dev_ans, fp)
    try:
        for target_dir in ['cnn_model3', 'rnn_model3']:
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


