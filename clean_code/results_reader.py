import os
from model import cbow, image_layers
import torch
import matplotlib.pyplot as plt


for root, dirs, files in os.walk('../data_aws/'):
    if root in ['../data_aws/img_data', '../data_aws/text_data', '../data_aws/', '../data_aws/text_data/Easy', '../data_aws/text_data/Hard']:
        continue
    print(root)
    model_best = torch.load(os.path.join(root, 'checkpoint_best.pth.tar'))
    model_latest = torch.load(os.path.join(root, 'checkpoint_latest.pth.tar'))

    # fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 3, 1)
    plt.plot(model_latest['model'].losses, 'o-')
    plt.title(root)
    plt.ylabel('losses')

    plt.subplot(2, 3, 2)
    plt.plot(model_latest['model'].losses_pos, 'o-')
    plt.ylabel('losses_pos')

    plt.subplot(2, 3, 3)
    plt.plot(model_latest['model'].losses_test, 'o-')
    plt.ylabel('losses_test')

    plt.subplot(2, 3, 4)
    plt.plot(model_latest['model'].top1s, 'o-')
    plt.ylabel('top1')

    plt.subplot(2, 3, 5)
    plt.plot(model_latest['model'].top3s, 'o-')
    plt.ylabel('top3')

    plt.subplot(2, 3, 6)
    plt.plot(model_latest['model'].top5s, 'o-')
    plt.ylabel('top5')

    print(model_latest['model'].top1s)
    print(model_latest['model'].top3s)
    print(model_latest['model'].top5s)

    # plt.show()
    plt.savefig(os.path.join(root, 'graphs.png'))
    plt.clf()
