import time
import torch
from torch.autograd import Variable

import torch.nn.functional as F

import numpy as np
def test(model, rnn2, image_layer, loader, config):
    CUDA = config.CUDA
    cosine_similarity = config.cosine_similarity

    start = time.time()
    model.eval()
    test_loss = 0
    N = 0
    top1 = 0
    top3 = 0
    top5 = 0
    hist = torch.zeros(10)
    for j, batch in enumerate(loader):
        sizes = batch['size']
        text, img_feat, target = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target'])
        if len(text.size()) < 2:
            text = text.view(1, -1)
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()
        text_prediction = model(text)
        if config.sequential:
            maxlen = torch.max(sizes)
            reshaped_tensor = Variable(torch.zeros(sizes.size(0), maxlen, config.emb_size).cuda())
            if CUDA:
                reshaped_tensor = reshaped_tensor.cuda()
            tot_idx = 0
            for i, size in enumerate(sizes):
                reshaped_tensor[i, :, :] = F.pad(text_prediction[tot_idx:tot_idx + size], (0, 0, 0, maxlen - size), 'constant', 0)
                tot_idx += size
            text_prediction = rnn2(reshaped_tensor, sizes=sizes)

        if len(text_prediction.size()) < 2:
            text_prediction = text_prediction.view(1, -1)
        # device = 0
        # show_memusage(device=device)
        img_prediction = img_feat
        if image_layer != None:
            img_prediction = image_layer(img_feat)

        total_idx = 0
        img_total_idx = 0
        predictions = []


        if cosine_similarity:
            xp = torch.mm(text_prediction, img_prediction.t())
            xx = torch.sqrt(torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t())
            pp = torch.sqrt(torch.sum(img_prediction * img_prediction, 1).expand(xp.size()))
            xp = xp / (xx * pp)
        else:
            xp = -2 * torch.mm(text_prediction, img_prediction.t())
            xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
            xp += torch.sum(img_prediction * img_prediction, 1).expand(xp.size())

        for i, size in enumerate(sizes):
            if config.concat or config.sequential:
                if cosine_similarity:
                    scores = -xp[i, i * 10:i * 10 + 10].data
                else:
                    scores = (xp[i, i*10:i*10+10]).data
                    # scores = (1 / img_prediction.size(1)) * (xp[i, i*10:i*10+10]).data
            else:
                if cosine_similarity:
                    # scores = xp[total_idx:total_idx + size, img_total_idx:img_total_idx+img_prediction.size(0)].sum(0).data
                    scores = -xp[total_idx:total_idx + size, i * 10:i * 10 + 10].sum(0).data
                else:
                # scores = torch.sqrt(xp[total_idx:total_idx+size, img_total_idx:img_total_idx+img_prediction.size(0)]).sum(0).data
                #     scores = (1 / img_prediction.size(1)) * (xp[total_idx:total_idx+size, i*10:i*10+10]).sum(0).data
                    scores = (xp[total_idx:total_idx+size, i*10:i*10+10]).sum(0).data
            target = target.squeeze()
            # scores = (1+scores) / 2
            test_loss += scores[target.data[i]]
            top1 += 1 if target.data[i] in scores.topk(1, largest=False)[1] else 0
            top3 += 1 if target.data[i] in scores.topk(3, largest=False)[1] else 0
            top5 += 1 if target.data[i] in scores.topk(5, largest=False)[1] else 0
            predictions.append(scores)
            N += 1
            total_idx += size

            # hist[img_prediction.size(0)-1] += 1
            img_total_idx += img_prediction.size(0)


    # print(hist)
    # test_loss
    top1 /= N
    top3 /= N
    top5 /= N
    return test_loss, test_loss / N, top1, top3, top5

