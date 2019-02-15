
'''
Check the label distribution of data
'''
from collections import defaultdict

def read_dataset(filename):
  freq = [0] * 16
  total_num = 0
  t2i = defaultdict(lambda: len(t2i))
  tag_set = set()
  with open(filename, "r") as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      tag_set.add(tag)
      tag = t2i[tag]
      freq[tag] += 1
      total_num += 1
  for tag_idx in range(16):
      print('tag_idx: {}, {:.4f}%'.format(tag_idx, freq[tag_idx] * 100/total_num))
  print(tag_set)


read_dataset("../topicclass/dev.txt")
