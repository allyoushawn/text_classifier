import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pdb

class CNNModel(nn.Module):
    def __init__(self, n_words, n_tags, n_embeds, n_hid, embed_matrix):
        super(CNNModel, self).__init__()
        self.n_words = n_words
        self.n_tags = n_tags
        self.n_hid = n_hid
        self.n_embeds = n_embeds

        self.embeds = nn.Embedding(n_words, n_embeds)
        embed_matrix = torch.tensor(embed_matrix)
        #self.embeds.load_state_dict({'weight': embed_matrix})
        #self.embeds.weight.requires_grad = False
        self.conv2_bn = nn.BatchNorm2d(n_hid)
        self.word_embed_bn = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, n_hid, kernel_size=(2, n_embeds), padding=(1, 0))
        self.conv2 = nn.Conv2d(1, n_hid, kernel_size=(3, n_embeds), padding=(2, 0))
        self.conv3 = nn.Conv2d(1, n_hid, kernel_size=(4, n_embeds), padding=(3, 0))
        self.fc1 = nn.Linear(3 * n_hid, 1024)
        self.fc2 = nn.Linear(1024, n_tags)
        self.fc_drop = nn.Dropout(p=0.5)


    def forward(self, words, lengths=None, hidden=None):
        word_embeds = self.embeds(words)
        word_embeds = word_embeds.unsqueeze(dim=1)
        word_embeds = self.word_embed_bn(word_embeds)

        x1 = self.conv1(word_embeds)
        x1 = self.conv2_bn(x1)
        x1 = torch.relu(x1)
        x1 = F.max_pool2d(x1, kernel_size=x1.shape[2:])
        #Global max pooling over while sequence
        x2 = self.conv2(word_embeds)
        x2 = self.conv2_bn(x2)
        x2 = torch.relu(x2)
        x2 = F.max_pool2d(x2, kernel_size=x2.shape[2:])
        x3 = self.conv3(word_embeds)
        x3 = self.conv2_bn(x3)
        x3 = torch.relu(x3)
        x3 = F.max_pool2d(x3, kernel_size=x3.shape[2:])
        x = torch.cat((x1,x2, x3), dim=1)
        x = x.squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x;


    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self, os.path.join(path, 'model.pth'))


    @staticmethod
    def open(path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model

    def load(self, path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model


class LSTMModel(nn.Module):
    def __init__(self, n_words, n_tags, n_embeds, n_hid, embed_matrix):
        super(LSTMModel, self).__init__()
        self.n_words = n_words
        self.n_tags = n_tags
        self.n_hid = n_hid
        self.n_embeds = n_embeds

        # Using pretrained embedding
        self.embeds = nn.Embedding(n_words, n_embeds)
        embed_matrix = torch.tensor(embed_matrix)
        #self.embeds.load_state_dict({'weight': embed_matrix})


        self.rnn1 = nn.LSTM(n_embeds, n_hid, 2, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(n_hid, 1024)
        self.fc2 = nn.Linear(1024, n_tags)
        self.fc_drop = nn.Dropout(p=0.5)


    def forward(self, words, lengths, hidden=None):
        x = self.embeds(words)
        #x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.rnn1(x, hidden)
        #x = self.lockdrop(x, 0.5)
        #x, _ = self.rnn2(x, hidden)
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
            len(lengths), x.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if x.is_cuda:
            idx = idx.cuda(x.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = x.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        x = last_output
        #x = torch.cat((x[:,0, :], last_output), dim=1) # For bi-LSTM

        x = self.fc1(x)
        x = self.fc_drop(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self, os.path.join(path, 'model.pth'))


    @staticmethod
    def open(path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model

    def load(self, path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model


class EXAMModel(nn.Module):
    def __init__(self, n_words, n_tags, n_embeds, n_hid, embed_matrix):
        super(EXAMModel, self).__init__()
        self.n_words = n_words
        self.n_tags = n_tags
        self.n_hid = n_hid
        self.n_embeds = n_embeds

        # Using pretrained embedding
        self.embeds = nn.Embedding(n_words, n_embeds)
        embed_matrix = torch.tensor(embed_matrix)
        self.embeds.load_state_dict({'weight': embed_matrix})


        self.rnn1 = nn.LSTM(n_embeds, n_hid, 2, batch_first=True, bidirectional=False)
        #self.rnn2 = nn.LSTM(n_hid, n_hid, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(n_hid, n_tags)
        self.fc_drop = nn.Dropout(p=0.5)
        self.interact_mat = nn.Parameter(torch.randn(n_tags, n_hid))
        self.W1 = nn.Parameter(torch.randn(1, 100))
        self.b1 = nn.Parameter(torch.randn(100))
        self.W2 = nn.Parameter(torch.randn(100, 1))


    def forward(self, words, lengths, hidden=None):
        x = self.embeds(words)
        #x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.rnn1(x, hidden)
        #x = self.lockdrop(x, 0.5)
        #x, _ = self.rnn2(x, hidden)
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(-1, self.n_hid)
        aggregat = torch.mm(self.interact_mat, torch.t(x))
        aggregat = aggregat.view(self.n_tags, T, N)
        aggregat = aggregat.permute(0, 2, 1)
        aggregat = aggregat.contiguous().view(-1, T)
        W1 = self.W1.repeat(T, 1)
        b1 = self.b1.repeat(self.n_tags * N, 1)
        aggregat = torch.mm(aggregat, W1) + b1
        A = F.relu(aggregat)
        logits = torch.mm(A, self.W2)
        logits = logits.view(self.n_tags, N)
        logits = logits.permute(1, 0)



        return logits

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self, os.path.join(path, 'model.pth'))


    @staticmethod
    def open(path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model

    def load(self, path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(
            os.path.join(path, 'model.pth'), map_location=device)
        return model
