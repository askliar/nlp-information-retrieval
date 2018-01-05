import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CBOW(nn.Module):
    def __init__(self, vocab_size, img_feat_size, target_size, CUDA):
        super(CBOW, self).__init__()
        self.losses = []
        self.losses_avg = []
        self.losses_pos = []
        self.losses_pos_avg = []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.losses_test = []
        self.losses_test_avg = []
        self.embeddings = torch.nn.Embedding(vocab_size, img_feat_size, padding_idx=0)
        self.proj = nn.Linear(img_feat_size, target_size)
        self.CUDA = CUDA
        self.img_feat_size = img_feat_size

    def forward(self, input_text):
        input_text = input_text.view(input_text.size(0), -1)
        x = self.embeddings(input_text)
        x = torch.sum(x, 1)
        # for loop
            # if sizes is not None:
            #     question_sets_amount = sizes.size(0)
            #     x_temp = Variable(torch.zeros((question_sets_amount, self.img_feat_size)))
            #     if self.CUDA:
            #         x_temp = x_temp.cuda()
            #     idx = 0
            #     for i in range(question_sets_amount):
            #         x_temp[i] = torch.sum(x[idx:idx + sizes[i], :], 0)
            #         idx += sizes[i]
            #     x = x_temp
        x = self.proj(F.relu(x))
        return x
