import sys
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import pdb




def balance_train_data(target_dir, filename):
    freq = [0] * 16
    total_num = 0
    t2i = defaultdict(lambda: len(t2i))
    increased_tag = set([3, 4, 7, 10, 11, 12, 14, 15])
    try:
      basename = filename.split('/')[-1]
      op_f = open(target_dir + '/' + basename, 'w')
    except:
      raise IOError('Directory: {} not exist.'.format(target_dir))


    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            tag = t2i[tag]
            if tag in increased_tag:
                op_f.write(line)
                op_f.write(line)
                op_f.write(line)
                op_f.write(line)
                op_f.write(line)
            else:
                op_f.write(line)
    op_f.close()


def remove_function_words(target_dir, filename):
    try:
        basename = filename.split('/')[-1]
        op_f = open(target_dir + '/' + basename, 'w')
    except:
        raise IOError('Directory: {} not exist.'.format(target_dir))
    stop_wrd_set = set(stopwords.words('english'))
    with open(filename, 'r') as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            op_str = tag + ' ||| '
            for word in words.split():
                if word not in stop_wrd_set:
                    op_str += word + ' '
            op_str = op_str.rstrip()
            op_f.write(op_str + '\n')

    op_f.close()




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Should give a target directory')
        quit()
    target_dir = sys.argv[1]
    tr_file = '../topicclass/train.txt'
    dev_file = '../topicclass/dev.txt'
    test_file = '../topicclass/test.txt'

    #balance_train_data(target_dir, tr_file)
    remove_function_words(target_dir, tr_file)
    remove_function_words(target_dir, dev_file)
    remove_function_words(target_dir, test_file)
