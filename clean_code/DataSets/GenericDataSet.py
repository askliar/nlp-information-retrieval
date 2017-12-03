import os
import pickle
import json

import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


stop = set(stopwords.words('english'))
stop.remove('no')
stop.update(['?', '\'s', ',', '!', '.', '-', ':', '_', ''])

stemmer = SnowballStemmer("english")

class GenericDataSet(Dataset):

    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, questions = False, stem=True, stopwords=False, stop_vocab = None, normalize=True, debug=False):

        self.img_features, self.mean, self.std, self.visual_feat_mapping = self.load_image_data(img_feat_file,
                                                                                                img_map_file)
        self.vocab = vocab
        if stopwords:
            self.stop_vocab = stop_vocab
        retrieve = os.path.isfile(pickle_file)
        if retrieve:
            self.data = pickle.load(open(pickle_file, 'rb'))
        else:
            df = pd.read_json(json_file).T.sort_index()
            if debug:
                df = df.head(5)

            self.data = self.preprocess_data(df, vocab, questions, stem, stopwords, stop_vocab)
            if not debug:
                pickle.dump(self.data, open(pickle_file, 'wb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tensor, img_id, target = self.data[idx]

        h5_id = self.visual_feat_mapping[str(img_id)]
        img_feat = torch.FloatTensor(self.img_features[h5_id])

        sample = (text_tensor, img_feat, target)
        return sample

    def preprocess_data(self, df,  vocab, questions, stem, stopwords, stop_vocab):
        data_storage = []
        for row in df.itertuples():
            text = row.question
            if questions:
                text_tensor, target = self.convert_to_int(text, vocab)
            else:
                target = 1
                text_tensor = self.convert_to_int(text, vocab)

            img_id = row.target_img_id
            target = torch.FloatTensor(target)
            text_tensor = torch.LongTensor(text_tensor)
            data_storage.append((text_tensor, img_id, target))
        return data_storage

    def text2int(self, text, vocab):
        return [vocab[word] for word in text]

    def preprocess_text(self, text, vocab, stem, stopwords, stop_vocab):
        if stopwords:
            preprocessed_text = tuple([(lambda x: stemmer.stem(x) if stem else x)(word)
                                       for word in word_tokenize(text.lower())
                                       if word not in stop_vocab]
                                      )
        else:
            preprocessed_text = tuple([(lambda x: stemmer.stem(x) if stem else x)(word)
                                       for word in word_tokenize(text.lower())])

        return preprocessed_text

    def load_image_data(self, img_feat_file, img_map_file):
        img_features = np.asarray(h5py.File(img_feat_file, 'r')['img_features'])
        with open(img_map_file, 'r') as f:
            visual_feat_mapping = json.load(f)['IR_imgid2id']
        mean = img_features.mean(axis=0)
        std = img_features.std(axis=0)
        img_features = (img_features - mean) / std
        return img_features, mean, std, visual_feat_mapping

class QuestionsDataSet(GenericDataSet):

    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file, captions,
                 vocab, stem=True, stopwords=False, stop_vocab = None, normalize=True, debug=False):
        questions = True
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, questions, stem, stopwords, stop_vocab, normalize, debug)


    def convert_to_int(self, text, vocab, stem, stopwords, stop_vocab):
        questions = [sentence for sentences in text for sentence in sentences]
        questions_int = []
        answers_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                question_preprocessed = super.preprocess_text(question, vocab, stem, stopwords, stop_vocab)
                question_int = super.text2int(self, question_preprocessed, vocab)
                augmented_question_int = question_int[:-1] + [(vocab['no'] if answer == 'yes' else vocab['yes'])]
                answer_int = 1 if answer == 'yes' else -1
                augmented_answer_int = -answer_int
                answers_int.extend([answer_int, augmented_answer_int])
                questions_int.extend([question_int, augmented_question_int])
        return self.pad_text(questions_int), answers_int

class CaptionsDataSet(GenericDataSet):

    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file, captions,
                 vocab, stem=True, stopwords=False, stop_vocab = None, normalize=True, debug=False):

        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, text, vocab, stem, stopwords, stop_vocab):
        caption_preprocessed = super.preprocess_text(text, vocab, stem, stopwords, stop_vocab)
        return super.text2int(self, caption_preprocessed, vocab)



# class TestDataSet(Dataset):
#     def __init__(self, json_file, pickle_file, img_features, visual_feat_mapping,
#                  mean, std, vocab, stem=False, stopwords=False, stop_vocab = None,
#                  normalize=True, debug=False):
#         retrieve = os.path.isfile(pickle_file)
#         self.img_features = img_features
#         self.visual_feat_mapping = visual_feat_mapping
#         self.mean = mean
#         self.std = std
#         self.vocab = vocab
#         self.stem = stem
#         self.stopwords = stopwords
#         if stopwords:
#             self.stop_vocab = stop_vocab
#
#         if retrieve:
#             self.data = pickle.load(open(pickle_file, 'rb'))
#         else:
#             df = pd.read_json(json_file).T.sort_index()
#
#             if debug:
#                 df = df.head(5)
#
#             self.data = self.preprocess_data(df)
#
#             if not debug:
#                 pickle.dump(self.data, open(pickle_file, 'wb'))
#
#     def preprocess_data(self, df):
#         data_storage = []
#         for row in df.itertuples():
#             caption = row.caption
#             dialog = row.dialog
#             imgids = row.img_list
#             target = row.target
#             resulting_dialog = self.questions2int(dialog, self.vocab)
#             resulting_caption = self.caption2int(caption, self.vocab)
#             text_tensor = torch.FloatTensor(pad_text([resulting_caption] + resulting_dialog))
#             target = torch.FloatTensor(target)
#             data_storage.append((text_tensor, imgids, target))
#         return data_storage
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data[idx]
#         text_tensor, imgids, target = row
#         imgs = torch.FloatTensor(self.imgids2imgs(imgids))
#         return (text_tensor, imgs, target)
#
#     def caption2int(self, caption, vocab):
#         if stopwords:
#             processed_caption = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
#                                        for word in word_tokenize(caption.lower())
#                                        if word not in self.stop_vocab]
#                                       )
#         else:
#             processed_caption = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
#                                        for word in word_tokenize(caption.lower())])
#
#         processed_caption = text2int(processed_caption, vocab)
#         return processed_caption
#
#     def questions2int(self, questions, vocab):
#         max_len = 0
#         questions = [sentence for sentences in questions for sentence in sentences]
#         questions_int = []
#         for question in questions:
#             answer = question.split('?')[1].strip().lower()
#             is_binary = answer == 'yes' or answer == 'no'
#             if is_binary:
#                 # processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
#                 #                                                         for word in word_tokenize(question.lower())
#                 #                                                         if not stopwords or word not in self.stop_vocab]
#                 #                                                        )
#                 if stopwords:
#                     processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
#                                             for word in word_tokenize(question.lower())
#                                             if word not in self.stop_vocab]
#                                            )
#                 else:
#                     processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
#                                                 for word in word_tokenize(question.lower())])
#                 question_int = text2int(processed_question, vocab)
#                 questions_int.append(question_int)
#         return pad_text(questions_int)
#
#     def imgids2imgs(self, img_id_arr):
#         imgs = []
#         for img_id in img_id_arr:
#             h5id = self.visual_feat_mapping[str(img_id)]
#             img = self.img_features[h5id]
#             img = (img - self.mean) / self.std
#             imgs.append(img)
#         return imgs


