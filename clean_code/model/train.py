import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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

    train_loss = 0
    positive = 0
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
        device = 0
        show_memusage(device=device)
        optimizer.zero_grad()
        text_prediction = model(text, sizes)
        img_prediction = img_feat
        if image_layer != None:
            img_prediction = image_layer(img_feat)

        if config.sequential:
            pass

        else:

            # st = time.time()
            if config.collapse:
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

            if config.collapse:
                loss = distances.mean()
            else:
                loss = (distances * target).mean()

        train_loss += loss.data[0]
        # print(train_loss[0])
        positive += distances.mean().data[0]  # loss.data[0]
        device = 0
        show_memusage(device=device)
        loss.backward()
        optimizer.step()

    return train_loss, positive