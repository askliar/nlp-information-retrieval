import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable



def train(model, optimizer, loader, config):
    model.train()
    CUDA = config.CUDA

    train_loss = 0
    positive = 0
    j = 0
    for batch in loader:
        startb = time.time()
        # if j > 10:
        #     break
        # j += 1
        text, img_feat, target, sizes = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target']), \
                                        Variable(batch['size'])
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()

        optimizer.zero_grad()
        text_prediction = model(text)

        # st = time.time()
        idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes.data[x])])
        # print('timeee -- ', time.time() - st)
        if CUDA:
            idx = idx.cuda()
        if config.cosine_similarity:
            distances = F.cosine_similarity(text_prediction, img_feat[idx])
        else:
            distances = F.pairwise_distance(text_prediction, img_feat[idx])
        loss = (distances * target).mean()

        train_loss += loss.data[0]
        # print(train_loss[0])
        positive += distances.mean().data[0]  # loss.data[0]

        loss.backward()
        optimizer.step()

    return train_loss, positive