# @profile
# def preprocess_test_data(complexity, split='val', stem=True, stopwords=True):
#     csv_file = '{}/IR_{}_{}.json'.format(complexity.capitalize(), split, complexity)
#     csv_path = os.path.join(text_data, csv_file)
#
#     df = pd.read_json(csv_path).T.sort_index()  # .head(5)
#     df_questions = df.copy()
#     s = df_questions.apply(lambda x: pd.Series(x['dialog']), axis=1) \
#         .stack() \
#         .reset_index(level=1, drop=True)
#     q_result = s.apply(lambda q:
#                        (lambda x=q[0].split('?')[1].strip().lower(): x if x == 'yes' or x == 'no' else '')())
#     q_result = q_result.reset_index(drop=True)
#     q_bool = s.apply(lambda q:
#                      (lambda x=q[0].split('?')[1].strip().lower(): True if x == 'yes' or x == 'no' else False)())
#     s.name = 'question'
#     q_result = pd.Series(q_result[q_bool.reset_index(drop=True)] \
#                          .reset_index(drop=True)) \
#         .apply(lambda x: 1 if x == 'yes' else -1)
#     q_result.name = 'question_result'
#     df_questions = df_questions.drop('dialog', axis=1).join(s)[q_bool].reset_index(drop=True) \
#         .join(q_result).dropna() \
#         .reset_index(drop=True).drop('target', axis=1)
#     df_questions.question = df_questions.question.apply(lambda x: x[0])
#     df_questions.question = df_questions.question.apply(lambda sentence:
#                                                         tuple(
#                                                             [(lambda x: stemmer.stem(x) if stem else x)(i)
#                                                              for i in word_tokenize(sentence.lower())
#                                                              if i not in stop or not stopwords]
#                                                         )
#                                                         )
#     df.img_list = df.img_list.apply(lambda arr: tuple(arr))
#     df_questions = df_questions[['question', 'img_list', 'target_img_id']]
#
#     df_captions = df.copy()
#     df_captions.caption = df_captions.caption.apply(lambda sentence:
#                                                     tuple(
#                                                         [(lambda x: stemmer.stem(x) if stem else x)(i)
#                                                          for i in word_tokenize(sentence.lower())
#                                                          if i not in stop or not stopwords]
#                                                     )
#                                                     )
#     df_captions = df_captions[['caption', 'img_list', 'target_img_id']].drop_duplicates().reset_index(drop=True)
#
#     s_captions = df_captions.apply(lambda x: pd.Series(x['img_list']), axis=1) \
#         .stack() \
#         .reset_index(level=1, drop=True)
#
#     s_questions = df_questions.apply(lambda x: pd.Series(x['img_list']), axis=1) \
#         .stack() \
#         .reset_index(level=1, drop=True)
#
#     s_captions.name = 'img_id'
#     s_questions.name = 'img_id'
#     df_captions = df_captions.drop('img_list', axis=1).join(s_captions).reset_index(drop=True)
#     df_captions = df_captions.rename(columns={'caption': 'text'})
#
#     df_questions = df_questions.drop(['img_list'], axis=1).join(s_questions).reset_index(drop=True)
#     df_questions = df_questions[['question', 'img_id', 'target_img_id']]
#     df_questions = df_questions.rename(columns={'question': 'text'})
#     df = df_captions.append(df_questions, ignore_index=True)
#
#     return df[['text', 'img_id', 'target_img_id']].sort_values(['target_img_id', 'text']).reset_index(drop=True)
#
# data_val = TextImageDataSet(split = 'val',  complexity = 'easy')
# data_test_captions = TextImageDataSet(split = 'test',  complexity = 'easy')
# data_test_questions = TextImageDataSet(split = 'test',  complexity = 'easy')
# dataloader_val_captions = DataLoader(data_val_captions, batch_size=1000,
#                         shuffle=False, num_workers=4)
# dataloader_val_questions = DataLoader(data_val_questions, batch_size=1000,
#                         shuffle=False, num_workers=4)
# dataloader_test_captions = DataLoader(data_test_captions, batch_size=1000,
#                         shuffle=False, num_workers=4)
# dataloader_test_questions = DataLoader(data_test_questions, batch_size=1000,
#                         shuffle=False, num_workers=4)

import json
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

DEBUG = False
CUDA = torch.cuda.is_available()

torch.manual_seed(1)

captions_batch_size = 128
questions_batch_size = 512

img_data = './data/img_data'
text_data = './data/text_data'
img_feat_file = 'IR_image_features.h5'
img_map_file = 'IR_img_features2id.json'
complexity = 'easy'

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
                                           '{}/IR_val1_{}.pkl'.format(complexity.capitalize(), complexity))
pickle_test_file = os.path.join(text_data,
                                           '{}/IR_test_{}.pkl'.format(complexity.capitalize(), complexity))


pickle_val_file_exists = os.path.isfile(pickle_val_file)
pickle_test_file_exists = os.path.isfile(pickle_test_file)

stop = set(stopwords.words('english'))
stop.remove('no')
stop.update(['?', '\'s', ',', '!', '.', '-', ':', '_', ''])

stemmer = SnowballStemmer("english")

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]

def read_dataset(caption_dataset, question_dataset, dictionary):
    captions = list(caption_dataset.df.caption)
    questions = list(question_dataset.df.question)
    all_sentences = captions + questions
    _ = [dictionary[word] for sentence in all_sentences for word in sentence]
    return dictionary

def pad_text(text):
    max_len = 0
    for sentence in text:
        max_len = max(max_len, len(sentence))
    resulting_questions = [sentence + ([0] * (max_len - len(sentence))) for sentence in text]
    return resulting_questions

class TrainDataSet(Dataset):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file, captions,
                 vocab, stem=True, stopwords=False, stop_vocab = None, normalize=True, debug=False):
        retrieve = os.path.isfile(pickle_file)
        self.captions = captions
        if retrieve:
            self.data = pickle.load(open(pickle_file, 'rb'))
        else:
            self.df = pd.read_json(json_file).T.sort_index()

            if debug:
                self.df = self.df.head(5)

            if captions:
                self.df = self.preprocess_captions(self.df, stem, stopwords, stop_vocab, debug)
            else:
                self.df = self.preprocess_questions(self.df, stem, stopwords, stop_vocab, debug)

            self.data = self.preprocess_data(self.df)

            if not debug:
                pickle.dump(self.data, open(pickle_file, 'wb'))

        self.vocab = vocab

        if stopwords:
            self.stop_vocab = stop_vocab
        self.img_features, self.mean, self.std, self.visual_feat_mapping = self.load_image_data(img_feat_file,
                                                                                                img_map_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_tensor, img_id, target = self.data[idx]

        h5_id = self.visual_feat_mapping[str(img_id)]
        img_feat = torch.FloatTensor(self.img_features[h5_id])

        sample = (text_tensor, img_feat, target)
        return sample

    def preprocess_data(self, df):
        data_storage = []
        for row in df.itertuples():
            if self.captions:
                text = row.caption
                target = 1
            else:
                text = row.question
                target = row.question_result
            img_id = row.target_img_id

            target = torch.FloatTensor(target)
            text_tensor = self.text2int(text, self.vocab)
            text_tensor = torch.LongTensor(text_tensor)

            data_storage.append((text_tensor, img_id, target))
        return data_storage

    def preprocess_captions(self, df, stem, stopwords, stop_vocab, debug):
        if debug:
            df = df.head(5)
        if stopwords:
            df.caption = df.caption.apply(lambda sentence:
                                          tuple(
                                              [(lambda x: stemmer.stem(x) if stem else x)(word)
                                               for word in word_tokenize(sentence.lower())
                                               if word not in stop_vocab ]
                                          )
                                          )
        else:
            df.caption = df.caption.apply(lambda sentence:
                                          tuple(
                                              [(lambda x: stemmer.stem(x) if stem else x)(word)
                                               for word in word_tokenize(sentence.lower())]
                                          )
                                          )
        return df[['caption', 'target_img_id']].drop_duplicates().reset_index(drop=True)

    def preprocess_questions(self, df, stem, stopwords, stop_vocab, debug):
        if debug:
            df = df.head(5)

        s = df.apply(lambda x: pd.Series(x['dialog']), axis=1) \
            .stack() \
            .reset_index(level=1, drop=True)
        q_result = s.apply(lambda q:
                           (lambda x=q[0].split('?')[1].strip().lower(): x if x == 'yes' or x == 'no' else '')())
        q_result = q_result.reset_index(drop=True)
        q_bool = s.apply(lambda q:
                         (lambda x=q[0].split('?')[1].strip().lower(): True if x == 'yes' or x == 'no' else False)())
        s.name = 'question'
        q_result = pd.Series(q_result[q_bool.reset_index(drop=True)] \
                             .reset_index(drop=True)) \
            .apply(lambda x: 1 if x == 'yes' else -1)
        q_result.name = 'question_result'
        df = df.drop('dialog', axis=1).join(s)[q_bool].reset_index(drop=True) \
            .join(q_result).dropna() \
            .reset_index(drop=True).drop('target', axis=1)
        df.question = df.question.apply(lambda x: x[0])
        if stopwords:
            df.question = df.question.apply(lambda sentence:
                                            tuple(
                                                [(lambda x: stemmer.stem(x) if stem else x)(word)
                                                 for word in word_tokenize(sentence.lower())
                                                 if word not in stop_vocab]
                                            )
                                            )
        else:
            df.question = df.question.apply(lambda sentence:
                                            tuple(
                                                [(lambda x: stemmer.stem(x) if stem else x)(word)
                                                 for word in word_tokenize(sentence.lower())]
                                            )
                                            )
        df_augmented = df.copy()
        df_augmented.question_result = -df_augmented.question_result
        df_augmented.question = df_augmented.question. \
            apply(lambda sent: sent[:-1] + ('yes',)
        if sent[-1] == 'no' else sent[:-1] + ('no',))
        df = df.append(df_augmented, ignore_index=True)
        return df[['question', 'target_img_id', 'question_result']]

    def load_image_data(self, img_feat_file, img_map_file):
        img_features = np.asarray(h5py.File(os.path.join(img_data, img_feat_file), 'r')['img_features'])
        with open(os.path.join(img_data, img_map_file), 'r') as f:
            visual_feat_mapping = json.load(f)['IR_imgid2id']
        mean = img_features.mean(axis=0)
        std = img_features.std(axis=0)
        img_features = (img_features - mean) / std
        return img_features, mean, std, visual_feat_mapping

class TestDataSet(Dataset):
    def __init__(self, json_file, pickle_file, img_features, visual_feat_mapping,
                 mean, std, vocab, stem=False, stopwords=False, stop_vocab = None,
                 normalize=True, debug=False):
        retrieve = os.path.isfile(pickle_file)
        self.img_features = img_features
        self.visual_feat_mapping = visual_feat_mapping
        self.mean = mean
        self.std = std
        self.vocab = vocab
        self.stem = stem
        self.stopwords = stopwords
        if stopwords:
            self.stop_vocab = stop_vocab

        if retrieve:
            self.data = pickle.load(open(pickle_file, 'rb'))
        else:
            df = pd.read_json(json_file).T.sort_index()

            if debug:
                df = df.head(5)

            self.data = self.preprocess_data(df)

            if not debug:
                pickle.dump(self.data, open(pickle_file, 'wb'))

    def preprocess_data(self, df):
        data_storage = []
        for row in df.itertuples():
            caption = row.caption
            dialog = row.dialog
            imgids = row.img_list
            target = row.target
            resulting_dialog = self.questions2int(dialog, self.vocab)
            resulting_caption = self.caption2int(caption, self.vocab)
            text_tensor = torch.FloatTensor(pad_text([resulting_caption] + resulting_dialog))
            target = torch.FloatTensor(target)
            data_storage.append((text_tensor, imgids, target))
        return data_storage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text_tensor, imgids, target = row
        imgs = torch.FloatTensor(self.imgids2imgs(imgids))
        return (text_tensor, imgs, target)

    def caption2int(self, caption, vocab):
        if stopwords:
            processed_caption = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                       for word in word_tokenize(caption.lower())
                                       if word not in self.stop_vocab]
                                      )
        else:
            processed_caption = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                       for word in word_tokenize(caption.lower())])

        processed_caption = text2int(processed_caption, vocab)
        return processed_caption

    def questions2int(self, questions, vocab):
        max_len = 0
        questions = [sentence for sentences in questions for sentence in sentences]
        questions_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                # processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                #                                                         for word in word_tokenize(question.lower())
                #                                                         if not stopwords or word not in self.stop_vocab]
                #                                                        )
                if stopwords:
                    processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                            for word in word_tokenize(question.lower())
                                            if word not in self.stop_vocab]
                                           )
                else:
                    processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                                for word in word_tokenize(question.lower())])
                question_int = text2int(processed_question, vocab)
                questions_int.append(question_int)
        return pad_text(questions_int)

    def imgids2imgs(self, img_id_arr):
        imgs = []
        for img_id in img_id_arr:
            h5id = self.visual_feat_mapping[str(img_id)]
            img = self.img_features[h5id]
            img = (img - self.mean) / self.std
            imgs.append(img)
        return imgs

class CBOW(nn.Module):
    def __init__(self, vocab_size, img_feat_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, img_feat_size)

    def forward(self, input_text):
        input_text = input_text.view(input_text.size(0), -1)
        x = self.embeddings(input_text)
        x = torch.sum(x, 1)
        return x

# load data
data_train_captions = TrainDataSet(json_file=json_train_file,
                                   pickle_file=pickle_captions_train_file,
                                   img_feat_file=img_feat_file,
                                   img_map_file=img_map_file,
                                   vocab=w2i, captions=True,
                                   stopwords=True, stop_vocab=stop,
                                   debug=DEBUG)

dataloader_train_captions = DataLoader(data_train_captions, batch_size=captions_batch_size,
                                       shuffle=True, num_workers=4, pin_memory=CUDA)

data_train_questions = TrainDataSet(json_file=json_train_file,
                                    pickle_file=pickle_questions_train_file,
                                    img_feat_file=img_feat_file,
                                    img_map_file=img_map_file,
                                    vocab=w2i, captions=False,
                                    stopwords=True, stop_vocab=stop,
                                    debug=DEBUG)

dataloader_train_questions = DataLoader(data_train_questions, batch_size=questions_batch_size,
                                        shuffle=True, num_workers=4, pin_memory=CUDA)

w2i = defaultdict(lambda: UNK,
                  read_dataset(data_train_captions, data_train_questions, w2i))

data_val = TestDataSet(json_file=json_val_file,
                       img_features=data_train_questions.img_features,
                       pickle_file=pickle_val_file,
                       visual_feat_mapping=data_train_questions.visual_feat_mapping,
                       mean=data_train_questions.mean,
                       std=data_train_questions.std,
                       vocab=w2i, stem=True,
                       stopwords=False, stop_vocab=None,
                       normalize=True, debug=DEBUG)

dataloader_val = DataLoader(data_val, num_workers=4, pin_memory=CUDA)

data_test = TestDataSet(json_file=json_test_file,
                        pickle_file=pickle_test_file,
                        img_features=data_train_questions.img_features,
                        visual_feat_mapping=data_train_questions.visual_feat_mapping,
                        mean=data_train_questions.mean,
                        std=data_train_questions.std,
                        vocab=w2i, stem=True,
                        stopwords=True, stop_vocab=stop,
                        normalize=True, debug=DEBUG)

dataloader_test = DataLoader(data_test, num_workers=4, pin_memory=CUDA)

nwords = len(w2i)

model = CBOW(nwords, 2048)

print(model)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters())


# @profile
def train(epoch, model, dataloader, batch_size=10, cuda=False):
    model.train()

    text_list = []
    dsize = dataloader.batch_size
    img_feat_batched = torch.zeros(batch_size * dsize, 2048)
    target_batched = torch.zeros(batch_size * dsize)
    dataloader_length = len(dataloader)

    for idx, sample in enumerate(dataloader):

        text_data = pad_text(sample['text'], w2i)
        img_feat = sample['img_features']
        target = sample['target'].float()
        # print(type(img_feat))
        # if cuda:
        #     text_data, img_feat, target = text_data.cuda(), img_feat.cuda(), target.cuda()%

        text_list.append(text_data)
        img_feat_batched[(idx % batch_size) * dsize: (idx % batch_size) * dsize + dsize] = img_feat
        target_batched[(idx % batch_size) * dsize: (idx % batch_size) * dsize + dsize] = target

        if (idx + 1) % batch_size == 0 or idx == dataloader_length - 1:
            if idx == dataloader_length - 1:
                batch_size = idx % batch_size

            lengths = [x.size(1) for x in text_list]
            maxlen = max(lengths)

            text_data_batched = torch.zeros(batch_size * dsize, maxlen).long()

            for i in range(batch_size):
                text_data_batched[i * dsize:i * dsize + dsize, :lengths[i]] = text_list[i]

            text_data_batched = text_data_batched.cuda()
            img_feat_batched = img_feat_batched.cuda()
            target_batched = target_batched.cuda()

            for i in range(batch_size):
                text_data = text_data_batched[i * dsize:i * dsize + dsize, :lengths[i]]
                img_feat = img_feat_batched[i * dsize:i * dsize + dsize, :]
                target = target_batched[i * dsize:i * dsize + dsize]

                text_data, img_feat, target = Variable(text_data), Variable(img_feat), Variable(target)

                optimizer.zero_grad()
                output = model(text_data.contiguous())
                loss = torch.sum(target * F.pairwise_distance(output, img_feat).view(-1))
                batch_loss = loss / target.size(0)
                batch_loss.backward()
                optimizer.step()
            text_list = []
            # train_loss += loss.data[0]

        ###    train_loss /= len(dataloader.dataset)
        # print('\nTrain epoch: {} Average loss: {:.4f}\n'.format(
        #     epoch, train_loss[0]))

# @profile
def evaluate(model, data_train, row, cuda=False):
    sentence = ' '.join(row[0])
    img_id = row[1]
    mean = data_train.mean
    std = data_train.std

    h5_id = data_train.visual_feat_mapping[str(img_id)]
    img_feat = torch.FloatTensor(data_train.img_features[h5_id])

    # if cuda:
    #     model = model.cuda()

    text_data = pad_text([sentence], w2i)
    if cuda:
        text_data, img_feat = text_data.cuda(), img_feat.cuda()

    text_data, img_feat = Variable(text_data), Variable(img_feat)

    output = model(text_data)
    loss = F.pairwise_distance(output, img_feat.view(output.size(0), -1))
    return loss.data[0][0]


def k_most_accuracy(chunk, target, k):
    assert k <= len(chunk), 'k is bigger than the size of a chunk!'
    if target in list(chunk[:k]):
        return 1
    return 0


first = 0


# @profile
def test(complexity, split, model, data_train, num_of_imgs=10, stem=True, stopwords=True, cuda=False):
    model.eval()
    global first
    if first == 0:
        df = preprocess_test_data(complexity, split, stem, stopwords)
        first += 1

    loss = pd.Series(df.apply(lambda row: evaluate(model, data_train_captions, row, cuda),
                              axis=1))
    loss.name = 'loss'
    df = pd.concat([df, loss], axis=1)
    temp_df = df.drop_duplicates(['target_img_id', 'img_id']).reset_index(drop=True)[['target_img_id', 'img_id']]
    grouped_df = df.groupby(['target_img_id', 'img_id'], as_index=False).aggregate(np.mean)
    grouped_df = pd.merge(temp_df, grouped_df, on=['target_img_id', 'img_id'], how='inner').iloc[:, :3]
    k_1_acc = 0
    k_3_acc = 0
    k_5_acc = 0
    num_of_chunks = int(len(grouped_df) / num_of_imgs)
    for i in range(num_of_chunks):
        chunk = grouped_df.iloc[i * num_of_imgs:(i * num_of_imgs) + num_of_imgs, :]
        target = chunk.target_img_id.iloc[0]
        ordered = chunk.sort_values('loss', ascending=False).reset_index(drop=True).img_id
        k_1_acc += k_most_accuracy(ordered, target, 1)
        k_3_acc += k_most_accuracy(ordered, target, 3)
        k_5_acc += k_most_accuracy(ordered, target, 5)
    k_1_acc /= num_of_chunks
    k_3_acc /= num_of_chunks
    k_5_acc /= num_of_chunks

    print('\nTop-1 accuracy: {:.4f}\n'.format(k_1_acc))

    print('\nTop-3 accuracy: {:.4f}\n'.format(k_3_acc))

    print('\nTop-5 accuracy: {:.4f}\n'.format(k_5_acc))


import time


def f():
    for epoch in range(1, 2):
        start = time.time()
        train(epoch, model, dataloader_train_captions, captions_batch_size,
              cuda=torch.cuda.is_available())
        train(epoch, model, dataloader_train_questions, questions_batch_size,
              cuda=torch.cuda.is_available())
        test('easy', 'val', model, data_train_captions, stem=True, stopwords=True, cuda=torch.cuda.is_available())
        print('Time taken: {}\n'.format(time.time() - start))


f()
