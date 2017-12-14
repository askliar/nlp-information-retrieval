import os
import torch

from utilities.nltk_helpers import stop


class Config():
    def __init__(self, remove_nonbinary=True, include_captions=False, augment_binary=True, cosine_similarity=True):
        self.DEBUG = False
        self.CUDA = torch.cuda.is_available()

        self.captions_batch_size = 64
        self.questions_batch_size = 64
        self.val_batch_size = 8
        self.test_batch_size = 8
        self.stopwords = True
        self.stop = stop

        torch.manual_seed(42)

        self.img_data = 'data/img_data'
        self.text_data = 'data/text_data'
        self.complexity = 'easy'

        self.remove_nonbinary = remove_nonbinary
        self.remove_nonbinary_str = 'bin' if self.remove_nonbinary else 'all'
        self.augment_binary = augment_binary
        self.augment_binary_str = 'aug' if self.augment_binary else 'non_aug'
        self.include_captions = include_captions
        self.include_captions_str = 'cap' if self.include_captions else 'no_cap'

        self.cosine_similarity = cosine_similarity
        self.cosine_similarity_str = 'cosine' if self.cosine_similarity else 'euclidean'

        self.uid_str = str.format('{}_{}_{}_{}', self.remove_nonbinary_str, self.augment_binary_str,
                                  self.include_captions_str, self.cosine_similarity_str)
        self.pickle_uid_str = str.format('{}_{}_{}', self.remove_nonbinary_str, self.augment_binary_str,
                                  self.include_captions_str)
        self.img_feat_file = os.path.join(self.img_data, 'IR_image_features.h5')
        self.img_map_file = os.path.join(self.img_data, 'IR_img_features2id.json')

        self.json_train_file = os.path.join(self.text_data,
                                            '{}/IR_train_{}.json'.format(self.complexity.capitalize(), self.complexity))
        self.json_val_file = os.path.join(self.text_data,
                                          '{}/IR_val_{}.json'.format(self.complexity.capitalize(), self.complexity))
        self.json_test_file = os.path.join(self.text_data,
                                           '{}/IR_test_{}.json'.format(self.complexity.capitalize(), self.complexity))

        self.pickle_captions_train_file = os.path.join(self.text_data,
                                                       '{}/IR_train_{}_captions_{}.pkl'.format(
                                                           self.complexity.capitalize(),
                                                           self.complexity, self.pickle_uid_str))
        self.pickle_questions_train_file = os.path.join(self.text_data,
                                                        '{}/IR_train_{}_questions_{}.pkl'.format(
                                                            self.complexity.capitalize(),
                                                            self.complexity, self.pickle_uid_str))
        self.pickle_val_file = os.path.join(self.text_data,
                                            '{}/IR_val_{}_{}.pkl'.format(self.complexity.capitalize(),
                                                                         self.complexity, self.pickle_uid_str))
        self.pickle_test_file = os.path.join(self.text_data,
                                             '{}/IR_test_{}_{}.pkl'.format(self.complexity.capitalize(),
                                                                           self.complexity, self.pickle_uid_str))

        self.pickle_vocab_file = os.path.join(self.text_data,
                                              '{}/vocab.pkl'.format(self.complexity.capitalize(), self.complexity))

        self.pickle_histograms_file = self.pickle_test_file = os.path.join(self.text_data,
                                             '{}/histogram_{}_{}.pkl'.format(self.complexity.capitalize(),
                                                                           self.complexity, self.pickle_uid_str))

        # if not os.path.exists(self.uid_str):
        #     os.mkdir(self.uid_str)
        if not os.path.exists('data/' + self.uid_str):
            os.mkdir('data/' + self.uid_str)
