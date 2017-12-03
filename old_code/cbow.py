import torch
from torch import nn
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, img_feat_size, output_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim + img_feat_size, output_dim)
        self.linear2 = nn.Linear(output_dim, 1)

    def forward(self, input_text, input_img_feat):
        x = self.embeddings(input_text)
        x = torch.sum(x, 1)
        print(input_img_feat.size())
        x = torch.cat([x, input_img_feat])
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
        return x