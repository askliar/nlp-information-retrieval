# load data
from collections import defaultdict
from torch.utils.data import DataLoader

from clean_code.clean_model import UNK
from clean_code.datasets.test_dataset import TestDataSet
from clean_code.datasets.train_dataset import CaptionsDataSet, QuestionsDataSet
from clean_code.utilities.data_helpers import train_c_fn, c_fn


class DataLoaderFactory():
    def __init__(self, config):
        self.config = config

    def generate_captions_dataset(self, vocab):
        data_train_captions = CaptionsDataSet(json_file=self.config.json_train_file,
                                              pickle_file=self.config.pickle_captions_train_file,
                                              img_feat_file=self.config.img_feat_file,
                                              img_map_file=self.config.img_map_file,
                                              vocab=vocab, vocab_pickle_file=self.config.pickle_vocab_file,
                                              stopwords=self.config.stopwords, stop_vocab=self.config.stop,
                                              debug=self.config.DEBUG)

        dataloader_train_captions = DataLoader(data_train_captions, batch_size=self.config.captions_batch_size,
                                               shuffle=True, num_workers=4, pin_memory=self.config.CUDA,
                                               collate_fn=train_c_fn)
        return data_train_captions, dataloader_train_captions

    def generate_questions_dataset(self, vocab):
        data_train_questions = QuestionsDataSet(json_file=self.config.json_train_file,
                                                pickle_file=self.config.pickle_questions_train_file,
                                                img_feat_file=self.config.img_feat_file,
                                                img_map_file=self.config.img_map_file,
                                                vocab=vocab, vocab_pickle_file=self.config.pickle_vocab_file,
                                                stopwords=True,
                                                stop_vocab=self.config.stop, debug=self.config.DEBUG)

        dataloader_train_questions = DataLoader(data_train_questions, batch_size=self.config.questions_batch_size,
                                                shuffle=True, num_workers=4, pin_memory=self.config.CUDA,
                                                collate_fn=train_c_fn)
        return data_train_questions, dataloader_train_questions

    def generate_val_dataset(self, vocab):
        data_val = TestDataSet(json_file=self.config.json_val_file,
                               pickle_file=self.config.pickle_val_file,
                               img_feat_file=self.config.img_feat_file,
                               img_map_file=self.config.img_map_file,
                               vocab=vocab, vocab_pickle_file=self.config.pickle_vocab_file,
                               stopwords=True,
                               stop_vocab=self.config.stop, debug=self.config.DEBUG)

        dataloader_val = DataLoader(data_val, batch_size=self.config.questions_batch_size,
                                    shuffle=True, num_workers=4, pin_memory=self.config.CUDA,
                                    collate_fn=c_fn)
        return data_val, dataloader_val

    def generate_test_dataset(self, vocab):
        data_test = TestDataSet(json_file=self.config.json_test_file,
                                pickle_file=self.config.pickle_test_file,
                                img_feat_file=self.config.img_feat_file,
                                img_map_file=self.config.img_map_file,
                                vocab=vocab, vocab_pickle_file=self.config.pickle_vocab_file,
                                stopwords=True,
                                stop_vocab=self.config.stop, debug=self.config.DEBUG)

        dataloader_test = DataLoader(data_test, batch_size=self.config.questions_batch_size,
                                     shuffle=True, num_workers=4, pin_memory=self.config.CUDA,
                                     collate_fn=c_fn)
        return data_test, dataloader_test


