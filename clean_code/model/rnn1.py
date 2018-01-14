import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
class RNN1(nn.Module):
    def __init__(self, vocab_size, img_feat_size, target_size, CUDA):
        super(RNN1, self).__init__()
        self.losses = []
        self.losses_avg = []
        self.losses_pos = []
        self.losses_pos_avg = []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.losses_test = []
        self.losses_test_avg = []
        self.rnn = nn.LSTM(img_feat_size, img_feat_size)
        self.embeddings = nn.Embedding(vocab_size, img_feat_size, padding_idx=0)
        self.proj = nn.Linear(img_feat_size, target_size)
        self.CUDA = CUDA
        self.img_feat_size = img_feat_size
    # @profile
    def forward(self, input_text, hidden=None):
        # input_text = input_text.view(input_text.size(0), -1)
        x = self.embeddings(input_text)
        # y = x.view(x.size(1), x.size(0), -1)
        # y = x
        # lens = torch.LongTensor(np.count_nonzero(input_text.data.cpu().numpy(), axis=0))#.cuda()
        lens1 = torch.LongTensor(np.count_nonzero(input_text.data.cpu().numpy(), axis=1))#.cuda()
        if self.CUDA:
            lens1 = lens1.cuda()
        # print(lens, lens1, lens2)
        # print(y.size(), [a for a in lens1[:, 0]])

        lengths, ind = torch.sort(lens1, 0, descending=True)
        y = pack_padded_sequence(x[ind], list(lengths), batch_first=True)
        if hidden is None:
            hidden = Variable(torch.zeros(1, x.size(0), self.img_feat_size))
            cn = Variable(torch.zeros(1, x.size(0), self.img_feat_size))
            if self.CUDA:
                hidden = hidden.cuda()
                cn = cn.cuda()

        # _, hidden = self.rnn(y, hidden)
        _, (hidden, _) = self.rnn(y, (hidden, cn))
        # packed_hidden = packed_hidden[ind]
        # output, hidden = pad_packed_sequence(packed_out)
        # print(output)
        # hidden = torch.gather(1, )
        # unsorted = input_text.new(*input_text.size())
        # input_text.scatter_(0, ind, input_text)
        # seq_end_idx = Variable(lens1)

        # row_indices = torch.arange(0, x.size(0)).long().cuda()
        # hidden = out[seq_end_idx, row_indices, :]
        _, revert_ind = ind.sort()
        x = self.proj(F.relu(hidden.squeeze()[revert_ind]))
        # del ind, revert_ind, lengths, y, lens1, hidden
        # del hidden
        # torch.cuda.empty_cache()
        return x