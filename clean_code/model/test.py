import time
import torch
from torch.autograd import Variable

from clean_code.utilities.config import CUDA

def test(model, loader):
    start = time.time()
    model.eval()
    if CUDA:
        test_loss = torch.zeros(1).cuda()
    test_loss = 0
    test_loss2 = 0
    N = 0
    top1 = 0
    top12 = 0
    top3 = 0
    top32 = 0
    top5 = 0
    top52 = 0
    for j, batch in enumerate(loader):

        text, img_feat, target, sizes = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target']), \
                                        Variable(batch['size'])
        # torch.cuda.synchronize()
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()
        # text.requires_grad = False
        # img_feat.requires_grad = False
        # optimizer.zero_grad()
        text_prediction = model(text)
        total_idx = 0
        predictions = []
        predictions2 = []

        xp = -2 * torch.mm(text_prediction, img_feat.t())
        xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
        xp += torch.sum(img_feat * img_feat, 1).expand(xp.size())
        total_idx = 0
        for i, size in enumerate(sizes.data):
            scores = torch.sqrt(xp[total_idx:total_idx+size, i*10:i*10+10]).sum(0).data
            test_loss += scores[target.data[0]]
            top1 += 1 if target.data[0] in scores.topk(1)[1] else 0
            top3 += 1 if target.data[0] in scores.topk(3)[1] else 0
            top5 += 1 if target.data[0] in scores.topk(5)[1] else 0
            predictions.append(scores)


            # scores2 = torch.zeros(10)
            # for k in range(10):
            #     dist = F.pairwise_distance(text_prediction[total_idx: (total_idx + size)],
            #                                img_feat[i + k].view(1, -1).expand(size, 2048)).sum()
            #     scores2[k] = torch.mean(dist).data[0]
            # test_loss2 += scores[target.data[0]]
            # top12 += 1 if target.data[0] in scores.topk(1)[1] else 0
            # top32 += 1 if target.data[0] in scores.topk(3)[1] else 0
            # top52 += 1 if target.data[0] in scores.topk(5)[1] else 0
            # predictions2.append(scores2)
            N += 1
            total_idx += size

        # for i, size in enumerate(sizes.data):
        #     scores = torch.zeros(10)
        #     for k in range(10):
        #         dist = F.pairwise_distance(text_prediction[total_idx: (total_idx + size)],
        #                                    img_feat[i + k].view(1, -1).expand(size, 2048)).sum()
        #         scores[k] = torch.mean(dist).data[0]
        #     test_loss += scores[target.data[0]]
        #     top1 += 1 if target.data[0] in scores.topk(1)[1] else 0
        #     top3 += 1 if target.data[0] in scores.topk(3)[1] else 0
        #     top5 += 1 if target.data[0] in scores.topk(5)[1] else 0
        #     predictions.append(scores)
        #     N += 1
        #     total_idx += size
    test_loss /= N
    top1 /= N
    top3 /= N
    top5 /= N
    print(top1, top3, top5)
    print(test_loss)
    print(time.time() - start)