import os
import torch

from clean_code.utilities.nltk_helpers import stop


class Config():
    def __init__(self, remove_nonbinary = True, include_captions = False, augment_binary = True,):
        self.DEBUG = False
        self.CUDA = torch.cuda.is_available()

        self.captions_batch_size = 256
        self.questions_batch_size = 256

        self.stopwords = True
        self.stop = stop

        torch.manual_seed(42)

        self.img_data = '../data/img_data'
        self.text_data = '../data/text_data'
        self.complexity = 'easy'

        self.remove_nonbinary = remove_nonbinary
        self.remove_nonbinary_str = 'bin' if self.remove_nonbinary else 'all'
        self.augment_binary = augment_binary
        self.augment_binary_str = 'aug' if self.augment_binary else 'non_aug'
        self.include_captions = include_captions
        self.include_captions_str = '-' if self.include_captions else 'caps_excl'

        self.cosine_similarity = True

        self.img_feat_file = os.path.join(self.img_data, 'IR_image_features.h5')
        self.img_map_file = os.path.join(self.img_data, 'IR_img_features2id.json')

        self.json_train_file = os.path.join(self.text_data,
                                       '{}/IR_train_{}.json'.format(self.complexity.capitalize(), self.complexity))
        self.json_val_file = os.path.join(self.text_data,
                                     '{}/IR_val_{}.json'.format(self.complexity.capitalize(), self.complexity))
        self.json_test_file = os.path.join(self.text_data,
                                      '{}/IR_test_{}.json'.format(self.complexity.capitalize(), self.complexity))

        self.pickle_captions_train_file = os.path.join(self.text_data,
                                                  '{}/IR_train_{}_captions.pkl'.format(self.complexity.capitalize(),
                                                                                       self.complexity))
        self.pickle_questions_train_file = os.path.join(self.text_data,
                                                   '{}/IR_train_{}_questions.pkl'.format(self.complexity.capitalize(),
                                                                                         self.complexity))
        self.pickle_val_file = os.path.join(self.text_data,
                                       '{}/IR_val_{}.pkl'.format(self.complexity.capitalize(), self.complexity))
        self.pickle_test_file = os.path.join(self.text_data,
                                        '{}/IR_test_{}.pkl'.format(self.complexity.capitalize(), self.complexity))

        self.pickle_vocab_file = os.path.join(self.text_data,
                                         '{}/vocab.pkl'.format(self.complexity.capitalize(), self.complexity))
