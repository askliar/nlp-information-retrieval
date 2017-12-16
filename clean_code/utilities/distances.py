import time
import torch
from torch.autograd import Variable


def cosine_similarity(text_prediction, img_prediction):
    if len(img_prediction.size()) == 1:
        img_prediction = img_prediction.view(1, -1)
    xp = torch.mm(text_prediction, img_prediction.t())
    xx = torch.sqrt(torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t())
    pp = torch.sqrt(torch.sum(img_prediction * img_prediction, 1).expand(xp.size()))
    res = xp / (xx * pp)
    return res

def mean_squared_error(text_prediction, img_prediction):
    if len(img_prediction.size()) == 1:
        img_prediction = img_prediction.view(1, -1)
    xp = -2 * torch.mm(text_prediction, img_prediction.t())
    xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
    xp += torch.sum(img_prediction * img_prediction, 1).expand(xp.size())
    return xp