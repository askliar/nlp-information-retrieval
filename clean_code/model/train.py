import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from clean_code.utilities.config import CUDA


def train(model, optimizer, loader, epochs=3):
    model.train()
    losses = []
    if CUDA:
        model.cuda()

    for e in range(epochs):
        start = time.time()

        train_loss = torch.zeros(1)
        train_loss_pos = torch.zeros(1)
        if CUDA:
            train_loss = train_loss.cuda()
            train_loss_pos = train_loss.cuda()

        j = 0
        for batch in loader:
            startb = time.time()
            if j > 10:
                break
            j += 1
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

            distances = F.pairwise_distance(text_prediction, img_feat[idx])
            loss = (distances * target).mean()
            # loss = F.cosine_similarity(text_prediction, img_feat[torch.cuda.LongTensor(idx)]).sum()
            train_loss += loss.data[0]
            train_loss_pos += distances.mean().data[0]  # loss.data[0]
            loss.backward()
            optimizer.step()

        if e < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.float_power(5, 1 / 5)

        # losslist.append(train_loss)
        print(train_loss[0])
        losses.append(train_loss[0])
        print('time epoch ', e, ' -> ', time.time() - start)

        if e % 11 == 0:
            torch.save(model, 'cap_checkpoint_' + str(e))
    pickle.dump(losses, open('cap_losses', 'wb'))
    import matplotlib.pyplot as plt
    plt.plot(losses)
