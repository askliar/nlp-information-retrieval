from collections import defaultdict
import torch.optim as optim

from datasets.dataloaders import DataLoaderFactory
from model.cbow import CBOW
from model.rnn1 import RNN1
from model.checkpointer import save_checkpoint
from model.train import train
from model.test import test
from utilities.config import Config
import numpy as np
import torch
import time
import pickle
from model.image_layers import MLP1, MLP2
from utilities.data_helpers import plot_histogram

from utilities.helper_functions import str2bool
from model.rnn2 import RNN2
import itertools
"""Application entry point."""
import argparse


def main():
    """Main entry point of application."""
    parser = argparse.ArgumentParser(
        description='Parser for training parameters.'
    )
    parser.add_argument(
        '--onlybin',
        help='Exclude non-binary questions',
        type=str,
        default='True'
    )
    parser.add_argument(
        '--captions',
        help='Include captions.',
        type=str,
        default='False'
    )
    parser.add_argument(
        '--augment',
        help='Augment binary questions',
        type=str,
        default='True'
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs',
        type=int,
        default=50
    )
    parser.add_argument(
        '--cosine',
        help='Whether to use cosine similarity or euclidean distance',
        type=str,
        default='True'
    )
    parser.add_argument(
        '--image_layer',
        help='which image layer to use',
        type=str,
        default='None'
    )
    parser.add_argument(
        '--projection',
        help='whether use CBOW or RNN for word concatenation',
        type=str,
        default='CBOW'
    )
    parser.add_argument(
        '--sequential',
        help='use CBOW or RNN for word concatenation',
        type=str,
        default='False' 
    )
    parser.add_argument(
        '--concat',
        help='whether to consider all text as one',
        type=str,
        default='False'  
    )
    parser.add_argument(
        '--batch_size',
        help='batch size',
        type=int,
        default=64  
    )
    parser.add_argument(
        '--test_batch_size',
        help='test_batch size',
        type=int,
        default=32  
    )
    parser.add_argument(
        '--lr',
        help='learning rate',
        type=float,
        default=0.0001  
    )
    parser.add_argument(
        '--target_space',
        help='dimensionality of target space',
        type=float,
        default=True
    )
    parser.add_argument(
        '--complexity',
        help='choose between easy and hard dataset',
        type=str,
        default='easy'
    )
    parser.add_argument(
        '--negbackprop',
        help='choose between easy and hard dataset',
        type=str,
        default='True'
    )

    args = parser.parse_args()
    onlybin = str2bool(args.onlybin)
    captions = str2bool(args.captions)
    augment = str2bool(args.augment)
    epochs = args.epochs
    image_layer = args.image_layer
    cosine_similarity = str2bool(args.cosine)
    projection = args.projection
    sequential = str2bool(args.sequential)
    concat = str2bool(args.concat)
    batch_size = int(args.batch_size)
    test_batch_size = int(args.test_batch_size)
    complexity = args.complexity
    learning_rate = float(args.lr)
    neg_backprop = str2bool(args.negbackprop)

    config = Config(include_captions=captions, remove_nonbinary=onlybin, augment_binary=augment,
                    cosine_similarity=cosine_similarity, image_layer=image_layer, projection=projection,
                    sequential=sequential, concat=concat, batch_size=batch_size, test_batch_size=test_batch_size,
                    complexity=complexity, learning_rate=learning_rate, neg_backprop=neg_backprop)

    factory = DataLoaderFactory(config)

    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]
    best_top1 = 0.0

    # captions_dataset, captions_dataloader = factory.generate_captions_dataset(w2i)
    questions_dataset, questions_dataloader = factory.generate_questions_dataset(w2i)

    # print(questions_dataset.sentences_histograms)
    # plot_histogram(questions_dataset.sentences_histograms)
    # print(captions_dataset.sentences_histograms)
    # plot_histogram(captions_dataset.sentences_histograms)

    w2i = defaultdict(lambda: UNK, questions_dataset.vocab)

    val_dataset, val_dataloader = factory.generate_val_dataset(w2i)
    # test_dataset, test_dataloader = factory.generate_test_dataset(w2i)
    emb_size = config.emb_size
    if image_layer != 'None' or config.sequential:
        num_feat = emb_size
    else:
        num_feat = 2048

    CUDA = config.CUDA
    if config.projection == 'CBOW':
        model = CBOW(vocab_size=len(w2i), img_feat_size=emb_size, target_size=num_feat, CUDA=CUDA)
    elif config.projection == 'RNN1':
        model = RNN1(vocab_size=len(w2i), img_feat_size=emb_size, target_size=num_feat, CUDA=CUDA)

    if config.sequential:
        rnn2 = RNN2(num_feat, emb_size, 2048, CUDA)
        if CUDA:
            rnn2 = rnn2.cuda()
    else:
        rnn2 = None

    image_layer = None
    if config.image_layer == 'mlp1':
        image_layer = MLP1(input_size=2048, output_size=num_feat)
    elif config.image_layer == 'mlp2':
        image_layer = MLP2(input_size=2048, hidden_size=emb_size, output_size=num_feat)



    lr_mult = 1 if config.cosine_similarity else 10
    if rnn2 != None:
        optimizer = optim.Adam(
                    itertools.chain(model.parameters(), rnn2.parameters()),
                    lr=config.learning_rate * lr_mult
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate * lr_mult
        )
    # model = torch.load('../results/checkpoint_22')
    # test(model, dataloader_val)

    if CUDA:
        model.cuda()
        if image_layer != None:
            image_layer = image_layer.cuda()
    losses = []
    losses_pos = []
    losses_avg = []
    losses_pos_avg = []

    for e in range(epochs):
        print('start of epoch ', e, 'for uid ', config.uid_str)

        start = time.time()
        train_loss, train_loss_pos, train_loss_avg, train_loss_pos_avg, \
            = train(model, rnn2, image_layer, optimizer, questions_dataloader, config)

        if config.captions_batch_size > 256:
            if e < 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= np.float_power(16, 1 / 10)
        if config.captions_batch_size == 512:
            if e < 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= np.float_power(8, 1 / 10)
        if e > 25 and e % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print('losses: ', train_loss_avg, train_loss_pos_avg)
        losses.append(train_loss)
        losses_avg.append(train_loss_avg)
        losses_pos.append(train_loss_pos)
        losses_pos_avg.append(train_loss_pos_avg)
        model.losses = losses
        model.losses_avg = losses_avg
        model.losses_pos = losses_pos
        model.losses_pos_avg = losses_pos_avg
        print('time epoch ', e, ' -> ', time.time() - start)

        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        test_time = time.time()
        test_loss, test_loss_avg, top1, top3, top5 = \
            test(model, rnn2, image_layer, val_dataloader, config)
        model.losses_test.append(test_loss)
        model.losses_test_avg.append(test_loss_avg)
        model.top1s.append(top1)
        model.top3s.append(top3)
        model.top5s.append(top5)
        print('top k accuracies: ', top1, top3, top5)
        print('test loss: ', test_loss_avg)
        print('test time: ', time.time() - test_time)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'epoch': e + 1,
            'model': model,
            'optimizer': optimizer,
        }, is_best, config)

        plot_list = [model.losses, model.losses_avg, model.losses_pos, model.losses_pos_avg,
                     model.losses_test, model.losses_test_avg, model.top1s, model.top3s, model.top5s]
        pickle.dump(plot_list, open(str.format('data/{}/plot_list.pkl', config.uid_str), 'wb'))
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    import matplotlib.pyplot as plt
    # plt.plot(losses)
    # plt.show()


if __name__ == '__main__':
    main()
