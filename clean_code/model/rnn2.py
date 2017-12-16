import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, target_size, CUDA):
        super(RNN2, self).__init__()
        self.losses = []
        self.losses_pos = []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.losses_test = []
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu')
        self.proj = nn.Linear(hidden_size, target_size)
        self.CUDA = CUDA
        self.input_size = input_size

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = Variable(torch.zeros(1, x.size(1), self.input_size))
            if self.CUDA:

                hidden = hidden.cuda()

        # if sizes is not None:
        #     question_sets_amount = sizes.size(0)
        #     x_temp = Variable(torch.zeros(question_sets_amount))
        #     if self.CUDA:
        #         x_temp = x_temp.cuda()
        #     idx = 0
        #     for i in range(question_sets_amount):
        #         x_temp[i] = torch.sum(x[idx:idx + sizes[i]])
        #         idx += sizes[i]
        #     x = x_temp
        out, hidden = self.rnn(x, hidden)
        x = self.proj(out)
        return x