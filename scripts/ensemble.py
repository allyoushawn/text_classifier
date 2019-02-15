#!/usr/bin/env python3
import pickle
import numpy as np

if __name__ == '__main__':
    hit = 0
    init_flag = True
    scores_collection = None
    with open('cnn_model1/dev_ans.pkl', 'rb') as fp:
        dev_ans = pickle.load(fp)
    for target_dir in ['rnn_model3', 'rnn_model6', 'rnn_model9', 'rnn_model12', 'cnn_model3', 'cnn_model6', 'cnn_model7', 'cnn_model11']:
    #for target_dir in ['rnn_model3', 'rnn_model6', 'rnn_model9', 'rnn_model12']:
    #for target_dir in ['cnn_model3', 'cnn_model6', 'cnn_model7', 'cnn_model11']:
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

