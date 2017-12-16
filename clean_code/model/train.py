import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.rnn2 import RNN2

try:
    import gpustat
except ImportError:
    raise ImportError("pip install gpustat")


def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def train(model, image_layer, optimizer, loader, config):
    model.train()
    CUDA = config.CUDA
    rnn2 = None
    if config.sequential:
        rnn2 = RNN2(2048, 2048, 2048, CUDA)

    train_loss = 0
    positive = 0
    n_batches = 0
    j = 0
    for batch in loader:
        startb = time.time()
        # if j > 2:
        #     break
        # j += 1

        # exclude sizes->Variable conversion, because we don't use it for backpropagation
        sizes = batch['size']
        text, img_feat, target = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target'])
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()
        # device = 0
        # show_memusage(device=device)
        optimizer.zero_grad()
        # if config.concat:
        #     text_prediction = model(text)
        # else:
        text_prediction = model(text)
        img_prediction = img_feat
        if image_layer != None:
            img_prediction = image_layer(img_feat)

        if config.sequential:
            print('we')
            maxlen = torch.max(sizes)
            print(sizes.size())
            reshaped_tensor = Variable(torch.zeros(maxlen, sizes.size(0), 2048).cuda())
            if CUDA:
                reshaped_tensor = reshaped_tensor.cuda()
            tot_idx = 0
            for i, size in enumerate(sizes):
                # a = F.pad(text_prediction[tot_idx:tot_idx + size], (0, 0, 0, maxlen - size), 'constant', 0)
                # print(a.size())
                print(reshaped_tensor.size())
                reshaped_tensor[:, i, :] = F.pad(text_prediction[tot_idx:tot_idx + size], (0, 0, 0, maxlen - size), 'constant', 0)
                tot_idx += size
            out, hidden = rnn2(reshaped_tensor)

        else:
            # st = time.time()
            if config.concat:
                idx = torch.LongTensor([x for x in range(sizes.size(0))])
            else:
                idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes[x])])
            # print('timeee -- ', time.time() - st)
            if CUDA:
                idx = idx.cuda()
            if config.cosine_similarity:
                distances = F.cosine_similarity(text_prediction, img_prediction[idx])
            else:
                distances = F.pairwise_distance(text_prediction, img_prediction[idx])

            if config.concat:
                loss = distances.mean()
            else:
                loss = (distances * target).mean()
        n_batches += 1
        train_loss += loss.data[0]
        # print(train_loss[0])
        positive += distances.mean().data[0]  # loss.data[0]
        # device = 0
        # show_memusage(device=device)
        loss.backward()
        optimizer.step()

    return train_loss, positive, train_loss / n_batches, positive / n_batches