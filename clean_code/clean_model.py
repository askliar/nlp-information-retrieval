from collections import defaultdict
import torch.optim as optim

from clean_code.datasets.dataloaders import DataLoaderFactory
from clean_code.model.cbow import CBOW
from clean_code.model.train import train
from clean_code.model.test import test
from clean_code.utilities.config import Config


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
        type = bool
    )
    parser.add_argument(
        '--captions',
        help='Include captions.',
        type=bool
    )
    parser.add_argument(
        '--augment',
        help='Augment binary questions',
        type=bool
    )
    args = parser.parse_args()
    onlybin = args.onlybin
    captions = args.captions
    augment = args.augment

    config = Config(include_captions=captions,remove_nonbinary=onlybin,augment_binary=augment)
    factory = DataLoaderFactory(config)

    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]

    captions_dataset, captions_dataloader = factory.generate_captions_dataset(w2i)
    questions_dataset, questions_dataloader = factory.generate_questions_dataset(w2i)

    w2i = defaultdict(lambda: UNK, questions_dataset.vocab)

    val_dataset, val_dataloader = factory.generate_val_dataset(w2i)
    test_dataset, test_dataloader = factory.generate_test_dataset(w2i)

    model = CBOW(vocab_size=len(w2i), img_feat_size=2048)
    optimizer = optim.Adam(model.parameters(), lr=0.0005 / 10)

    train(model, optimizer, questions_dataloader, 100)
    test(model, val_dataloader)



if __name__ == '__main__':

    main()
