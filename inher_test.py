import json
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import torch
from torch.utils.data import Dataset, DataLoader

DEBUG = False
CUDA = torch.cuda.is_available()

torch.manual_seed(1)

captions_batch_size = 128
questions_batch_size = 512

img_data = './data/img_data'
text_data = './data/text_data'
complexity = 'easy'

img_feat_file = os.path.join(img_data, 'IR_image_features.h5')
img_map_file = os.path.join(img_data, 'IR_img_features2id.json')

json_train_file = os.path.join(text_data,
                               '{}/IR_train_{}.json'.format(complexity.capitalize(), complexity))
json_val_file = os.path.join(text_data,
                             '{}/IR_val_{}.json'.format(complexity.capitalize(), complexity))
json_test_file = os.path.join(text_data,
                              '{}/IR_test_{}.json'.format(complexity.capitalize(), complexity))

pickle_captions_train_file = os.path.join(text_data,
                                          '{}/IR_train_{}_captions.pkl'.format(complexity.capitalize(), complexity))
pickle_questions_train_file = os.path.join(text_data,
                                           '{}/IR_train_{}_questions.pkl'.format(complexity.capitalize(), complexity))
pickle_val_file = os.path.join(text_data,
                               '{}/IR_val_{}.pkl'.format(complexity.capitalize(), complexity))
pickle_test_file = os.path.join(text_data,
                                '{}/IR_test_{}.pkl'.format(complexity.capitalize(), complexity))

pickle_vocab_file = os.path.join(text_data,
                                 '{}/vocab.pkl'.format(complexity.capitalize(), complexity))

stop = set(stopwords.words('english'))
stop.remove('no')
stop.update(['?', '\'s', ',', '!', '.', '-', ':', '_', ''])

stemmer = SnowballStemmer("english")

w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]


class GenericDataSet(Dataset):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, questions=False, test=False,
                 stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        self.img_features, self.mean, self.std, self.visual_feat_mapping = self.load_image_data(img_feat_file,
                                                                                                img_map_file)
        self.vocab = vocab
        if stopwords:
            self.stop_vocab = stop_vocab
        retrieve = os.path.isfile(pickle_file)
        if retrieve:
            with open(pickle_file, 'rb') as data_pickle, open(vocab_pickle_file, 'rb') as vocab_pickle:
                self.data = pickle.load(data_pickle)
                self.vocab = pickle.load(vocab_pickle)
            data_pickle.close()
            vocab_pickle.close()
        else:
            df = pd.read_json(json_file).T.sort_index()
            if debug:
                df = df.head(5)

            self.data = self.preprocess_data(df, questions, test, stem, stopwords, stop_vocab)
            if not debug:
                with open(pickle_file, 'wb') as data_pickle, open(vocab_pickle_file, 'wb') as vocab_pickle:
                    pickle.dump(self.data, data_pickle)
                    pickle.dump(dict(vocab), vocab_pickle)
                data_pickle.close()
                vocab_pickle.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # text_tensor = self.data['text'][idx]
        # img_id  = self.data['img_id'][idx]
        # target = self.data['target'][idx]
        text_tensor, img_id, target, text_size = self.data[idx]

        h5_id = self.visual_feat_mapping[str(img_id)]
        img_feat = torch.FloatTensor(self.img_features[h5_id])

        sample = {'text': text_tensor, 'img_id': img_feat, 'target': target, 'size': text_size}
        # sample = (text_tensor, img_feat, target)
        return sample

    def preprocess_data(self, df, questions, test, stem, stopwords, stop_vocab):
        data_storage = []
        # data_storage = {'text':[], 'img_id':[], 'target':[]}

        for row in df.itertuples():
            tup = self.convert_to_int(row, stem, stopwords, stop_vocab)
            if tup is not None:
                data_storage.append(tup)
                    # data_storage['text'].append(text_tensor)
                    # data_storage['img_id'].append(img_id)
                    # data_storage['target'].append(target)
        return data_storage

    def text2int(self, text):
        return [self.vocab[word] for word in text]

    def preprocess_text(self, text, stem, stopwords, stop_vocab):
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
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        questions = True
        test = False
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, questions, test, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        text = row.dialog
        questions = [sentence for sentences in text for sentence in sentences]
        questions_int = []
        answers_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                question_int = self.text2int(question_preprocessed)
                augmented_question_int = question_int[:-1] + [
                    (self.vocab['no'] if answer == 'yes' else self.vocab['yes'])]
                answer_int = 1 if answer == 'yes' else -1
                augmented_answer_int = -answer_int
                answers_int.extend([answer_int, augmented_answer_int])
                questions_int.extend([question_int, augmented_question_int])
        text_int = pad_text(questions_int)
        if len(text_int) > 0:
            img_id = row.target_img_id
            target = torch.FloatTensor(answers_int)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([text_tensor.size(0)])
            return (text_tensor, img_id, target, text_size)


class CaptionsDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        questions = False
        test = False
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, questions, test, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        text = row.caption
        caption_preprocessed = self.preprocess_text(text, stem, stopwords, stop_vocab)
        text_int = self.text2int(caption_preprocessed)
        target = [1]
        if len(text_int) > 0:
            img_id = row.target_img_id
            target = torch.FloatTensor(target)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([text_tensor.size(0)])
            return (text_tensor, img_id, target, text_size)


class TestDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        questions = False
        test = True
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, questions, test, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        question_int = self.convert_question_to_int(row.dialog, stem, stopwords, stop_vocab)
        caption_int = self.convert_caption_to_int(row.caption, stem, stopwords, stop_vocab)
        text_int = pad_text(question_int + [caption_int])
        img_id = row.img_list
        target = [row.target]
        target = torch.FloatTensor(target)
        text_tensor = torch.LongTensor(text_int)
        text_size = torch.LongTensor(text_tensor.size(0))
        return (text_tensor, img_id, target, text_size)

    def convert_caption_to_int(self, text, stem, stopwords, stop_vocab):
        caption_preprocessed = self.preprocess_text(text, stem, stopwords, stop_vocab)
        return self.text2int(caption_preprocessed)

    def convert_question_to_int(self, text, stem, stopwords, stop_vocab):
        questions = [sentence for sentences in text for sentence in sentences]
        questions_int = []
        answers_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                question_int = self.text2int(question_preprocessed)
                questions_int.append(question_int)
        return pad_text(questions_int)

    def imgids2imgs(self, img_id_arr):
        imgs = []
        for idx, img_id in enumerate(img_id_arr):
            h5id = self.visual_feat_mapping[str(img_id)]
            img = self.img_features[h5id]
            imgs.append(torch.FloatTensor((img - self.mean) / self.std))
        return imgs

    def __getitem__(self, idx):

        # text_tensor = self.data['text'][idx]
        # img_id  = self.data['img_id'][idx]
        # target = self.data['target'][idx]

        text_tensor, imgids, target, text_size = self.data[idx]
        imgs = torch.stack(self.imgids2imgs(imgids))

        sample = {'text': text_tensor, 'img_id': imgs, 'target': target, 'size': text_size}
        # sample = (text_tensor, imgs, target)
        return sample


def pad_text(text):
    max_len = 0
    for sentence in text:
        max_len = max(max_len, len(sentence))
    resulting_questions = [sentence + ([0] * (max_len - len(sentence))) for sentence in text]
    return resulting_questions


def pad_dict(data):
    max_len = 0
    if len(data[0]['text'].size()) > 1:
        for sample in data:
            if len(sample['text'].size()) > 1:
                max_len = max(max_len, sample['text'].size(1))
            else:
                max_len = max(max_len, len(sample['text']))
        for sample in data:
            if len(sample['text'].size()) > 1:
                padding_size = max_len - sample['text'].size(1)
                if padding_size != 0:
                    padding = torch.zeros((sample['text'].size(0), padding_size)).long()
                    sample['text'] = torch.cat((sample['text'], padding), 1)
            else:
                padding_size = max_len - len(sample['text'])
                if padding_size != 0:
                    sample['text'] = torch.cat((sample['text'], torch.zeros(padding_size).long()), 0)

    return data


def c_fn(batch):
    batch = pad_dict(batch)
    texts = []
    img_ids = []
    targets = []
    sizes = []
    for sample in batch:
        text_tensor = sample['text']
        img_id = sample['img_id']
        target = sample['target']
        size = sample['size']
        texts.append(text_tensor)
        img_ids.append(img_id)
        targets.append(target)
        sizes.append(size)
    text_tensor = torch.cat(texts, 0)
    img_id = torch.cat(img_ids, 0)
    target = torch.cat(targets, 0)
    size = torch.cat(sizes, 0)
    return {'text': text_tensor, 'img_id': img_id, 'target': target, 'size': size}


# load data
data_train_captions = CaptionsDataSet(json_file=json_train_file,
                                      pickle_file=pickle_captions_train_file,
                                      img_feat_file=img_feat_file,
                                      img_map_file=img_map_file,
                                      vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                      stopwords=True, stop_vocab=stop,
                                      debug=DEBUG)

dataloader_train_captions = DataLoader(data_train_captions, batch_size=captions_batch_size,
                                       shuffle=True, num_workers=4, pin_memory=CUDA,
                                       collate_fn=c_fn)

data_train_questions = QuestionsDataSet(json_file=json_train_file,
                                        pickle_file=pickle_questions_train_file,
                                        img_feat_file=img_feat_file,
                                        img_map_file=img_map_file,
                                        vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                        stopwords=True,
                                        stop_vocab=stop, debug=DEBUG)

dataloader_train_questions = DataLoader(data_train_questions, batch_size=questions_batch_size,
                                        shuffle=True, num_workers=4, pin_memory=CUDA,
                                        collate_fn=c_fn)

w2i = defaultdict(lambda: UNK, data_train_questions.vocab)

data_val_questions = TestDataSet(json_file=json_val_file,
                                 pickle_file=pickle_val_file,
                                 img_feat_file=img_feat_file,
                                 img_map_file=img_map_file,
                                 vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                 stopwords=True,
                                 stop_vocab=stop, debug=DEBUG)

dataloader_val = DataLoader(data_val_questions, batch_size=questions_batch_size,
                            shuffle=True, num_workers=4, pin_memory=CUDA,
                            collate_fn=c_fn)

data_test_questions = TestDataSet(json_file=json_test_file,
                                  pickle_file=pickle_test_file,
                                  img_feat_file=img_feat_file,
                                  img_map_file=img_map_file,
                                  vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                  stopwords=True,
                                  stop_vocab=stop, debug=DEBUG)

dataloader_test = DataLoader(data_test_questions, batch_size=questions_batch_size,
                             shuffle=True, num_workers=4, pin_memory=CUDA,
                             collate_fn=c_fn)
i = 0
for batch in dataloader_test:
    if i > 0:
        break
    else:
        i += 1
        print(batch)
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
