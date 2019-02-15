#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import random
import pickle
import pdb
import sys


def read_dataset(filename):
  w2i = defaultdict(lambda: len(w2i))
  t2i = defaultdict(lambda: len(t2i))
  PADDING = w2i["<PADDING>"]
  UNK = w2i["<unk>"]
  idx_wrd_set = set()
  idx_wrd_list = []

  for wrd in ["<PADDING>", "<unk>"]:
      idx = w2i[wrd]
      if (idx, wrd) not in idx_wrd_set:
          idx_wrd_set.add((idx, wrd))
          idx_wrd_list.append((idx, wrd))

  with open(filename, "r") as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      for x in words.split(" "):
          idx = w2i[x]
          if (idx, x) not in idx_wrd_set:
              idx_wrd_set.add((idx, x))
              idx_wrd_list.append((idx, x))

  return idx_wrd_list




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Should give a target directory')
        quit()
    target_dir = sys.argv[1]
    filename = "../topicclass/train.txt"
    idx_wrd_list = read_dataset(filename)
    dim = 300
    glove_file = "glove/glove.6B.{}d.txt".format(dim)
    glove_file = 'wiki-news-300d-1M.vec'


    op_f = open(target_dir + '/' + 'glove_oov_word.txt', 'w')
    op_f2 = open(target_dir + '/' + 'embedding.txt', 'w')
    word_embed = {}
    with open(glove_file) as f:
        for line in f.readlines():
            wrd = line.rstrip().split()[0]
            feats = line.rstrip().split()[1:]
            word_embed[wrd] = feats

    feats = []
    for tup in idx_wrd_list:
        idx, wrd = tup
        if wrd == '<PADDING>':
            op_str = "0 " * dim
            op_str = op_str.rstrip()
            op_f2.write(op_str + '\n')
            feat = [0.] * dim
            feats.append(feat)

        elif wrd == '<unk>' or wrd not in word_embed.keys():
            op_str = ''
            feat = []
            for _ in range(dim):
                rand_num = random.random()
                op_str += str(rand_num)
                op_str += ' '
                feat.append(rand_num)
            op_str = op_str.rstrip()
            op_f2.write(op_str + '\n')
            op_f.write(wrd + '\n')
            feats.append(feat)

        else:
            op_str = ''
            feat = []
            for feat_val in word_embed[wrd]:
                op_str += str(feat_val)
                op_str += ' '
                feat.append(feat_val)
            op_str = op_str.rstrip()
            op_f2.write(op_str + '\n')
            feats.append(feat)

    op_f2.close()
    op_f.close()
    feats_arr = np.array(feats, dtype=np.float32)
    with open(target_dir + '/' + 'embedding.pkl' , 'wb') as fp:
        pickle.dump(feats_arr, fp)
