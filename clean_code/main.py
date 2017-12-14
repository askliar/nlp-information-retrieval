from collections import defaultdict
import torch.optim as optim

from datasets.dataloaders import DataLoaderFactory
from model.cbow import CBOW
from model.checkpointer import save_checkpoint
from model.rnn_cbow import RNNCBOW
from model.train import train
from model.test import test
from utilities.config import Config

import torch
import time

from model.image_layers import MLP1, MLP2
from utilities.data_helpers import plot_histogram

from utilities.helper_functions import str2bool

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
        '--file',
        help='Path of a model',
        type=str,
        default=None
    )
    args = parser.parse_args()
    onlybin = str2bool(args.onlybin)
    captions = str2bool(args.captions)
    augment = str2bool(args.augment)
    epochs = args.epochs
    image_layer = args.image_layer
    cosine_similarity = str2bool(args.cosine)
    config = Config(include_captions=captions, remove_nonbinary=onlybin, augment_binary=augment,
                    cosine_similarity=cosine_similarity, image_layer=image_layer)

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

    CUDA = config.CUDA

    model = CBOW(vocab_size=len(w2i), img_feat_size=2048)
    image_layer = None
    if config.image_layer == 'mlp1':
        image_layer = MLP1(2048, 2048)
    elif config.image_layer == 'mlp2':
        image_layer = MLP2(2048, 2048, 2048)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # model = torch.load('../results/checkpoint_22')
    # test(model, dataloader_val)

    if CUDA:
        model.cuda()
        if image_layer != None:
            image_layer = image_layer.cuda()
    losses = []
    losses_pos = []

    for e in range(epochs):
        print('start of epoch ', e, 'for uid ', config.uid_str)

        start = time.time()
        train_loss, train_loss_pos = train(model, image_layer, optimizer, questions_dataloader, config)

        # if e < 5:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= np.float_power(5, 1 / 5)

        print('losses: ', train_loss, train_loss_pos)
        losses.append(train_loss)
        losses_pos.append(train_loss_pos)
        model.losses = losses
        model.losses_pos = losses_pos
        print('time epoch ', e, ' -> ', time.time() - start)

        test_time = time.time()
        test_loss, top1, top3, top5 = test(model, image_layer, val_dataloader, config)

        model.losses_test.append(test_loss)
        model.top1s.append(top1)
        model.top3s.append(top3)
        model.top5s.append(top5)
        print('top k accuracies: ', top1, top3, top5)
        print('test loss: ', test_loss)
        print('test time: ', time.time() - test_time)
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'epoch': e + 1,
            'model': model,
            'optimizer': optimizer,
        }, is_best, config)

    if CUDA:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
