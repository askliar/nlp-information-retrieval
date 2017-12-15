import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN1(nn.Module):
    def __init__(self, vocab_size, img_feat_size, target_size, CUDA):
        super(RNN1, self).__init__()
        self.losses = []
        self.losses_pos = []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.losses_test = []
        self.rnn = nn.RNN(img_feat_size, target_size, nonlinearity='relu')
        self.embeddings = nn.Embedding(vocab_size, img_feat_size, padding_idx=0)
        self.proj = nn.Linear(img_feat_size, target_size)
        self.CUDA = CUDA
        self.img_feat_size = img_feat_size

    def forward(self, input_text, sizes=None, hidden=None):
        input_text = input_text.view(input_text.size(0), -1)
        x = self.embeddings(input_text)
        y = x.view(x.size(1), x.size(0), -1)
        if hidden is None:
            hidden = Variable(torch.zeros(1, x.size(0), self.img_feat_size))
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
        out, hidden = self.rnn(y, hidden)
        x = self.proj(hidden.squeeze())
        return x