import os
from model import cbow, image_layers
import torch
import matplotlib.pyplot as plt
import pickle


for root, dirs, files in os.walk('../data_tel/'):
    if 'plot_list.pkl' in files:
        path = os.path.join(root, 'plot_list.pkl')
        # model_best =
        # model_latest = torch.load(os.path.join(root, 'checkpoint_latest.pth.tar'))
        [losses, losses_avg, losses_pos, losses_pos_avg,
                    losses_test, losses_test_avg,
                    top1s, top3s, top5s] = pickle.load(open(path, 'rb'))
        print(path)
        # aa = pickle.load(open(path, 'rb'))


        fig, ax = plt.subplots(nrows=2, ncols=2)

        plt.subplot(2, 3, 1)
        plt.plot(losses[10:], '-')
        plt.title(root)
        plt.ylabel('losses')

        plt.subplot(2, 3, 2)
        plt.plot(losses_pos[10:], '-')
        plt.ylabel('losses_pos')

        plt.subplot(2, 3, 3)
        plt.plot(losses_test[10:], '-')
        plt.ylabel('losses_test')

        plt.subplot(2, 3, 4)
        plt.plot(top1s[10:], '-')
        plt.ylabel('top1')

        plt.subplot(2, 3, 5)
        plt.plot(top3s[10:], '-')
        plt.ylabel('top3')

        plt.subplot(2, 3, 6)
        plt.plot(top5s[10:], '-')
        plt.ylabel('top5')

        print(top1s)
        print(top3s)
        print(top5s)

        # plt.show()
        plt.savefig(os.path.join(root, 'graphs.png'), dpi=600)
        plt.clf()
        plt.close()
