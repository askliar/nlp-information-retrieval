import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from utilities.distances import cosine_similarity, mean_squared_error
from torch.autograd import Variable

try:
    import gpustat
except ImportError:
    raise ImportError("pip install gpustat")


def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def train(model, rnn2, image_layer, optimizer, loader, config):
    model.train()
    CUDA = config.CUDA

    train_loss = 0.0
    positive = 0.0
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
        if not config.neg_backprop:
            # print('using absolute')
            target = torch.abs(target)
        if image_layer != None:
            print('using image layer')
            img_prediction = image_layer(img_feat)
        if config.sequential:
            maxlen = torch.max(sizes)
            reshaped_tensor = Variable(torch.zeros(sizes.size(0), maxlen, config.emb_size).cuda())
            if CUDA:
                reshaped_tensor = reshaped_tensor.cuda()
            tot_idx = 0
            for i, size in enumerate(sizes):
                reshaped_tensor[i, :, :] = F.pad(text_prediction[tot_idx:tot_idx + size], (0, 0, 0, maxlen - size), 'constant', 0)
                tot_idx += size
            out = rnn2(reshaped_tensor, sizes=sizes)
            if config.cosine_similarity:
                loss = - F.cosine_similarity(out, img_prediction).sum()
            else:
                loss = torch.pow(F.pairwise_distance(out, img_prediction), 2).sum()
            loss = loss / sizes.size(0)
            # print(loss)
            train_loss += loss.data[0]
            positive += 0.0

        else:
            if config.concat:
                idx = torch.LongTensor([x for x in range(sizes.size(0))])
            else:
                idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes[x])])
            # print('timeee -- ', time.time() - st)
            if CUDA:
                idx = idx.cuda()
            if config.cosine_similarity:
                distances = - F.cosine_similarity(text_prediction, img_prediction[idx])
            else:
                distances = torch.pow(F.pairwise_distance(text_prediction, img_prediction[idx]), 2)
                # distances = (1 / img_prediction.size(1)) * torch.pow(F.pairwise_distance(text_prediction, img_prediction[idx]), 2)
            if config.concat:
                loss = distances.mean()
            else:
                loss = (distances * target).mean()
                # losss = torch.nn.CosineSimilarity()
                # loss = losss(text_prediction, img_prediction[idx]).mean()
            train_loss += loss.data[0]
            positive += distances.mean().data[0]

        n_batches += 1
        loss.backward()
        optimizer.step()

    return train_loss, positive, train_loss / n_batches, positive / n_batches


# losses:  -96.9994621354133 206.41716721305846
# time epoch  0  ->  39.0826153755188
# top k accuracies:  0.1128 0.33 0.5412
# test loss:  7887.021224035644
# test time:  3.7204253673553467