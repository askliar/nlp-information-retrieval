import time
import torch
from torch.autograd import Variable

def test(model, loader, config):
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

        text, img_feat, target, sizes = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target']), \
                                        Variable(batch['size'])
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()
        text_prediction = model(text)
        total_idx = 0
        img_total_idx = 0
        predictions = []

        if cosine_similarity:
            xp = torch.mm(text_prediction, img_feat.t())
            xx = torch.sqrt(torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t())
            pp = torch.sqrt(torch.sum(img_feat * img_feat, 1).expand(xp.size()))
            xp = xp / (xx * pp)
        else:
            xp = -2 * torch.mm(text_prediction, img_feat.t())
            xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
            xp += torch.sum(img_feat * img_feat, 1).expand(xp.size())

        # print(text.size(), img_feat.size(), target.size(), sizes.size())
        for i, size in enumerate(sizes.data):
            # if img_feat.size(0) >= 5:
            # if cosine_similarity:
            #     scores = xp[total_idx:total_idx + size, img_total_idx:img_total_idx+img_feat.size(0)].sum(0).data
            # else:
            #     scores = torch.sqrt(xp[total_idx:total_idx+size, img_total_idx:img_total_idx+img_feat.size(0)]).sum(0).data
            # test_loss += scores[target.data[i]]
            # print(img_feat.size(0))
            # top1 += 1 if target.data[i] in scores.topk(1, largest=False)[1] else 0
            # top3 += 1 if target.data[i] in scores.topk(3, largest=False)[1] else 0
            # top5 += 1 if target.data[i] in scores.topk(5, largest=False)[1] else 0
            # predictions.append(scores)
            hist[img_feat.size(0)-1] += 1
            N += 1
            total_idx += size
            img_total_idx += img_feat.size(0)
    print(hist)
    test_loss /= N
    top1 /= N
    top3 /= N
    top5 /= N
    return test_loss, top1, top3, top5
