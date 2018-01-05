import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, target_size, CUDA):
        super(RNN2, self).__init__()
        self.losses = []
        self.losses_pos = []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.losses_test = []
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, target_size)
        self.input_size = input_size
        self.CUDA = CUDA
        self.img_feat_size = hidden_size

    def forward(self, input_text, hidden=None, sizes=None):
        if hidden is None:
            hidden = Variable(torch.zeros(1, input_text.size(0), self.img_feat_size))
            if self.CUDA:
                hidden = hidden.cuda()

        y = input_text
        if sizes is None:
            lens = torch.LongTensor(np.count_nonzero(input_text.data.cpu().numpy()[:, :, 0], axis=0)).cuda()
            lens1 = torch.LongTensor(np.count_nonzero(input_text.data.cpu().numpy()[:, :, 0], axis=1)).cuda()
        else:
            lens1 = sizes

        lengths, ind = torch.sort(lens1, 0, descending=True)
        y = pack_padded_sequence(y[ind], list(lengths), batch_first=True)

        out, hidden = self.rnn(y, hidden)

        _, revert_ind = ind.sort()
        x = self.proj(F.relu(hidden.squeeze()[revert_ind]))

        return x
    # def forward(self, x, hidden=None):
    #     if hidden is None:
    #         hidden = Variable(torch.zeros(1, x.size(1), self.input_size))
    #         if self.CUDA:
    #             hidden = hidden.cuda()
    #
    #
    #     # lengths, ind = torch.sort(lens1, 0, descending=True)
    #     # y = pack_padded_sequence(y[ind], list(lengths), batch_first=True)
    #     out, hidden = self.rnn(x, hidden)
    #     # _, revert_ind = ind.sort()
    #     x = self.proj(F.relu(out))
    #     return x