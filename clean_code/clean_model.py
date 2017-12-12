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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

DEBUG = False
CUDA = torch.cuda.is_available()

torch.manual_seed(1)

captions_batch_size = 256
questions_batch_size = 256
torch.manual_seed(42)
img_data = '../data/img_data'
text_data = '../data/text_data'
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


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, img_feat_size):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, img_feat_size, padding_idx=0)
        self.proj = nn.Linear(img_feat_size, img_feat_size)
    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)
    def forward(self, input_text):
        input_text = input_text.view(input_text.size(0), -1)
        x = self.embeddings(input_text)
        x = torch.sum(x, 1)
        x = self.proj(self.selu(x))
        return x


class GenericDataSet(Dataset):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False,
                 stop_vocab=None, normalize=True, debug=False):
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

            self.data = self.preprocess_data(df, stem, stopwords, stop_vocab)
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

        sample = {'text': text_tensor, 'img_feat': img_feat, 'target': target, 'size': text_size}
        # sample = (text_tensor, img_feat, target)
        return sample

    def preprocess_data(self, df, stem, stopwords, stop_vocab, augment_binary):
        data_storage = []
        # data_storage = {'text':[], 'img_id':[], 'target':[]}

        for row in df.itertuples():
            tup = self.convert_to_int(row, stem, stopwords, stop_vocab, augment_binary)
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
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True,
                 debug=False, augment_binary = True, remove_nonbinary = True, include_captions = True):
        self.augment_binary = augment_binary
        self.remove_nonbinary = remove_nonbinary
        self.include_captions = include_captions
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        questions_int, target_int = self.convert_question_to_int(row.dialog, stem, stopwords, stop_vocab)
        if self.include_captions:
            caption_int = self.convert_caption_to_int(row.caption, stem, stopwords, stop_vocab)
            text_int = pad_text(questions_int + [caption_int])
            target_int = target_int + [1]

        if len(questions_int) > 0:
            text_int = pad_text(questions_int)
            img_id = row.target_img_id
            target = torch.FloatTensor(target_int)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([text_tensor.size(0)])
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
                if self.augment_binary:
                    augmented_question_int = question_int[:-1] + [
                        (self.vocab['no'] if answer == 'yes' else self.vocab['yes'])]
                    answer_int = 1 if answer == 'yes' else -1
                    # BOOKMARK: orig questions
                    augmented_answer_int = -answer_int
                    answers_int.extend([answer_int, augmented_answer_int])
                    questions_int.extend([question_int, augmented_question_int])
                    # end of bookmark
                else:
                    answer_int = 1 if answer == 'yes' else -1
                    answers_int.append(answer_int)
                    questions_int.append(question_int)
            else:
                if self.remove_nonbinary:
                    continue
                else:
                    question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                    question_int = self.text2int(question_preprocessed)
                    answer_int = 1
                    answers_int.append(answer_int)
                    questions_int.append(question_int)
        return pad_text(questions_int), answers_int


class CaptionsDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab, augment_binary):
        text = row.caption
        caption_preprocessed = self.preprocess_text(text, stem, stopwords, stop_vocab)
        text_int = self.text2int(caption_preprocessed)
        target = [1]
        if len(text_int) > 0:
            img_id = row.target_img_id
            target = torch.FloatTensor(target)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([1])
            return (text_tensor, img_id, target, text_size)


class TestDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True,
                 debug=False, augment_binary = True, remove_nonbinary = True):
        self.augment_binary = augment_binary
        self.remove_nonbinary = remove_nonbinary
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab, augment_binary):
        questions_int = self.convert_question_to_int(row.dialog, stem, stopwords, stop_vocab)
        caption_int = self.convert_caption_to_int(row.caption, stem, stopwords, stop_vocab)
        text_int = pad_text(question_ints + [caption_int])
        img_id = row.img_list
        target = [row.target]
        if len(text_int) > 0:
            target = torch.LongTensor(target)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([text_tensor.size(0)])
            return (text_tensor, img_id, target, text_size)

    def convert_caption_to_int(self, text, stem, stopwords, stop_vocab):
        caption_preprocessed = self.preprocess_text(text, stem, stopwords, stop_vocab)
        return self.text2int(caption_preprocessed)

    def convert_question_to_int(self, text, stem, stopwords, stop_vocab):
        questions = [sentence for sentences in text for sentence in sentences]
        questions_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                question_int = self.text2int(question_preprocessed)
                if self.augment_binary:
                    augmented_question_int = question_int[:-1] + [
                        (self.vocab['no'] if answer == 'yes' else self.vocab['yes'])]
                    # BOOKMARK: orig questions
                    questions_int.extend([question_int, augmented_question_int])
                    # end of bookmark
                else:
                    questions_int.append(question_int)
            else:
                if self.remove_nonbinary:
                    continue
                else:
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

        sample = {'text': text_tensor, 'img_feat': imgs, 'target': target, 'size': text_size}
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


def pad_text_minbatches(texts, max_size):
    for idx, text in enumerate(texts):
        padding = torch.zeros((max_size - text.size(0), text.size(1))).long()
        texts[idx] = torch.cat((text, padding), 0)
    return texts


def train_c_fn(batch):
    batch = pad_dict(batch)
    texts = []
    img_ids = []
    targets = []
    sizes = []
    max_size = 0
    for sample in batch:
        text_tensor = sample['text']
        img_id = sample['img_feat']
        target = sample['target']
        size = sample['size']
        if len(text_tensor.size()) == 1:
            text_tensor = text_tensor.view(-1, text_tensor.size(0))
        texts.append(text_tensor)
        img_ids.append(img_id)
        targets.append(target)
        sizes.append(size)
        max_size = max(max_size, torch.max(size))
    # COMMENT FOR OLD
    # texts = pad_text_minbatches(texts, max_size)
    text_tensor = torch.cat(texts, 0)
    img_id = torch.stack(img_ids, 0)
    target = torch.cat(targets, 0)
    size = torch.cat(sizes, 0)

    return {'text': text_tensor, 'img_feat': img_id, 'target': target, 'size': size}


def c_fn(batch):
    batch = pad_dict(batch)
    texts = []
    img_ids = []
    targets = []
    sizes = []
    max_size = 0
    for sample in batch:
        text_tensor = sample['text']
        img_id = sample['img_feat']
        target = sample['target']
        size = sample['size']
        texts.append(text_tensor)
        img_ids.append(img_id)
        targets.append(target)
        sizes.append(size)
        max_size = max(max_size, torch.max(size))
    # COMMENT FOR OLD
    # texts = pad_text_minbatches(texts, max_size)
    text_tensor = torch.cat(texts, 0)
    img_id = torch.cat(img_ids, 0)
    target = torch.cat(targets, 0)
    size = torch.cat(sizes, 0)
    return {'text': text_tensor, 'img_feat': img_id, 'target': target, 'size': size}


if CUDA:
    try:
        import gpustat
    except ImportError:
        raise ImportError("pip install gpustat")


    def show_memusage(device=0):
        gpu_stats = gpustat.GPUStatCollection.new_query()
        item = gpu_stats.jsonify()["gpus"][device]
        print("{}/{}".format(item["memory.used"], item["memory.total"]))

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
                                       collate_fn=train_c_fn)

data_train_questions = QuestionsDataSet(json_file=json_train_file,
                                        pickle_file=pickle_questions_train_file,
                                        img_feat_file=img_feat_file,
                                        img_map_file=img_map_file,
                                        vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                        stopwords=True,
                                        stop_vocab=stop, debug=DEBUG)

dataloader_train_questions = DataLoader(data_train_questions, batch_size=questions_batch_size,
                                        shuffle=True, num_workers=4, pin_memory=CUDA,
                                        collate_fn=train_c_fn)

w2i = defaultdict(lambda: UNK, data_train_questions.vocab)

data_val = TestDataSet(json_file=json_val_file,
                                 pickle_file=pickle_val_file,
                                 img_feat_file=img_feat_file,
                                 img_map_file=img_map_file,
                                 vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                 stopwords=True,
                                 stop_vocab=stop, debug=DEBUG)

dataloader_val = DataLoader(data_val, batch_size=questions_batch_size,
                            shuffle=True, num_workers=4, pin_memory=CUDA,
                            collate_fn=c_fn)

data_test = TestDataSet(json_file=json_test_file,
                                  pickle_file=pickle_test_file,
                                  img_feat_file=img_feat_file,
                                  img_map_file=img_map_file,
                                  vocab=w2i, vocab_pickle_file=pickle_vocab_file,
                                  stopwords=True,
                                  stop_vocab=stop, debug=DEBUG)

dataloader_test = DataLoader(data_test, batch_size=questions_batch_size,
                             shuffle=True, num_workers=4, pin_memory=CUDA,
                             collate_fn=c_fn)




# @profile
def train(model, loader, epochs=3):
    model.train()
    losses = []
    if CUDA:
        model.cuda()

    for e in range(epochs):
        start = time.time()

        train_loss = torch.zeros(1)
        train_loss_pos = torch.zeros(1)
        if CUDA:
            train_loss = train_loss.cuda()
            train_loss_pos = train_loss.cuda()

        j = 0
        for batch in loader:
            startb = time.time()
            if j > 10:
                break
            j += 1
            text, img_feat, target, sizes = Variable(batch['text']), \
                                            Variable(batch['img_feat']), \
                                            Variable(batch['target']), \
                                            Variable(batch['size'])
            if CUDA:
                text, img_feat, target, sizes = text.cuda(), \
                                                img_feat.cuda(), \
                                                target.cuda(), \
                                                sizes.cuda()

            optimizer.zero_grad()
            text_prediction = model(text)

            # st = time.time()
            idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes.data[x])])
            # print('timeee -- ', time.time() - st)
            if CUDA:
                idx = idx.cuda()

            distances = F.pairwise_distance(text_prediction, img_feat[idx])
            loss = (distances * target).mean()
            # loss = F.cosine_similarity(text_prediction, img_feat[torch.cuda.LongTensor(idx)]).sum()
            train_loss += loss.data[0]
            train_loss_pos += distances.mean().data[0]  # loss.data[0]
            loss.backward()
            optimizer.step()


        if e < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.float_power(5, 1 / 5)

        # losslist.append(train_loss)
        print(train_loss[0])
        losses.append(train_loss[0])
        print('time epoch ', e, ' -> ', time.time() - start)

        if e % 11 == 0:
            torch.save(model, 'cap_checkpoint_' + str(e))
    pickle.dump(losses, open('cap_losses', 'wb'))
    import matplotlib.pyplot as plt
    plt.plot(losses)


def test(model, loader):
    start = time.time()
    model.eval()
    if CUDA:
        test_loss = torch.zeros(1).cuda()
    test_loss = 0
    test_loss2 = 0
    N = 0
    top1 = 0
    top12 = 0
    top3 = 0
    top32 = 0
    top5 = 0
    top52 = 0
    for j, batch in enumerate(loader):

        text, img_feat, target, sizes = Variable(batch['text']), \
                                        Variable(batch['img_feat']), \
                                        Variable(batch['target']), \
                                        Variable(batch['size'])
        # torch.cuda.synchronize()
        if CUDA:
            text, img_feat, target, sizes = text.cuda(), \
                                            img_feat.cuda(), \
                                            target.cuda(), \
                                            sizes.cuda()
        # text.requires_grad = False
        # img_feat.requires_grad = False
        # optimizer.zero_grad()
        text_prediction = model(text)
        total_idx = 0
        predictions = []
        predictions2 = []

        xp = -2 * torch.mm(text_prediction, img_feat.t())
        xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
        xp += torch.sum(img_feat * img_feat, 1).expand(xp.size())
        total_idx = 0
        for i, size in enumerate(sizes.data):
            scores = torch.sqrt(xp[total_idx:total_idx+size, i*10:i*10+10]).sum(0).data
            test_loss += scores[target.data[0]]
            top1 += 1 if target.data[0] in scores.topk(1)[1] else 0
            top3 += 1 if target.data[0] in scores.topk(3)[1] else 0
            top5 += 1 if target.data[0] in scores.topk(5)[1] else 0
            predictions.append(scores)


            # scores2 = torch.zeros(10)
            # for k in range(10):
            #     dist = F.pairwise_distance(text_prediction[total_idx: (total_idx + size)],
            #                                img_feat[i + k].view(1, -1).expand(size, 2048)).sum()
            #     scores2[k] = torch.mean(dist).data[0]
            # test_loss2 += scores[target.data[0]]
            # top12 += 1 if target.data[0] in scores.topk(1)[1] else 0
            # top32 += 1 if target.data[0] in scores.topk(3)[1] else 0
            # top52 += 1 if target.data[0] in scores.topk(5)[1] else 0
            # predictions2.append(scores2)
            N += 1
            total_idx += size

        # for i, size in enumerate(sizes.data):
        #     scores = torch.zeros(10)
        #     for k in range(10):
        #         dist = F.pairwise_distance(text_prediction[total_idx: (total_idx + size)],
        #                                    img_feat[i + k].view(1, -1).expand(size, 2048)).sum()
        #         scores[k] = torch.mean(dist).data[0]
        #     test_loss += scores[target.data[0]]
        #     top1 += 1 if target.data[0] in scores.topk(1)[1] else 0
        #     top3 += 1 if target.data[0] in scores.topk(3)[1] else 0
        #     top5 += 1 if target.data[0] in scores.topk(5)[1] else 0
        #     predictions.append(scores)
        #     N += 1
        #     total_idx += size
    test_loss /= N
    top1 /= N
    top3 /= N
    top5 /= N
    print(top1, top3, top5)
    print(test_loss)
    print(time.time() - start)

#
# 0.1 0.2966 0.5014
# 688.7573033172607
# 4.097585439682007

# 0.0982 0.2974 0.4992
# 690.3066700332641
# 13.765591859817505

# 0.1032 0.2974 0.502
# 690.4895692977906
# 13.727421283721924

# 0.1 0.2966 0.5014
# 145.83495834350586
# 4.106285572052002

# 0.1008 0.2896 0.4912
# 688.5925689743042
# 3.635680675506592

# 0.1014 0.2918 0.4956
# 689.1907331314087
# 3.5045695304870605

# 0.1042 0.2998 0.5046
# 691.4041340530396
# 13.347415447235107

model = CBOW(vocab_size=len(w2i), img_feat_size=2048)
optimizer = optim.Adam(model.parameters(), lr=0.0005 / 10)

train(model, dataloader_train_questions, 100)
test(model, dataloader_val)
import matplotlib.pyplot as plt
# plt.plot(losslist)
# plt.show()




#


# def glvq_costfun(self, x, batch_labels):
#     xp = -2 * torch.mm(x, self.prototypes.t())
#     xp += torch.sum(x * x, 1).expand(xp.t().size()).t()
#     xp += torch.sum(self.prototypes * self.prototypes, 1).expand(xp.size())
#     # xp = torch.norm()
#     e = xp.size()[0]
#     p = xp.size()[1]
#
#     xp2 = xp.clone()
#     xp2.data.scatter_(1, batch_labels.view(e, 1).data.long(), float("inf"))
#     _, arg_d1s = torch.min(xp, 1)
#     d1s = xp.gather(1, batch_labels.view(e, 1))
#
#     d2s, arg_d2s = torch.min(xp2, 1)
#     res = torch.sum((d1s - d2s) / (d1s + d2s))
#     return res, arg_d1s




#
# # @profile
# def train(model, loader, epochs=3):
#     model.train()
#     for e in range(epochs):
#
#         start = time.time()
#         if CUDA:
#             train_loss = torch.zeros(1).cuda()
#         j = 0
#         if CUDA:
#             model.cuda()
#         for batch in loader:
#             startb = time.time()
#             # torch.cuda.synchronize()
#             if j > 100000000:
#                 break
#             j += 1
#             text, img_feat, target, sizes = Variable(batch['text']), \
#                                             Variable(batch['img_feat']), \
#                                             Variable(batch['target']), \
#                                             Variable(batch['size'])
#             # torch.cuda.synchronize()
#             if CUDA:
#                 text, img_feat, target, sizes = text.cuda(), \
#                                                 img_feat.cuda(), \
#                                                 target.cuda(), \
#                                                 sizes.cuda()
#
#             optimizer.zero_grad()
#             print(text.size())
#             if text.size(1) > 25:
#                 continue
#             # show_memusage(device=device)
#
#             text_prediction = model(text)
#             # distances = F.pairwise_distance(text_prediction, img_feat)
#             # xp = -2 * torch.mm(text_prediction, img_feat.t())
#             # xp += torch.sum(text_prediction * text_prediction, 1).expand(xp.t().size()).t()
#             # xp += torch.sum(img_feat * img_feat, 1).expand(xp.size())
#
#             total_idx = 0
#             # exp1 = img_feat[0].view(1, -1).expand(sizes.data[0], 2048)
#
#             # startl = time.time()
#             # name = Variable(torch.zeros(1))
#             #
#             loss = Variable(torch.zeros(1))
#             if CUDA:
#             #     name = name.cuda()
#                 loss = loss.cuda()
#             # use gather instead of creating the exp1 vector
#             # for i, size in enumerate(sizes.data):
#             #     name += (F.cosine_similarity(text_prediction[total_idx: (total_idx + size)],
#             #                                           img_feat[i].view(1, -1).expand(size, 2048))).sum()
#             #     loss += torch.sqrt(xp[total_idx:total_idx+size, i]).sum()
#
#                 # exp1 = torch.cat((exp1, img_feat[i].view(1, -1).expand(size, 2048)), 0)
#                 # total_idx += size
#             st = time.time()
#             idx = torch.LongTensor([x for x in range(sizes.size(0)) for kk in range(sizes.data[x])])
#             print('timeee -- ', time.time() - st)
#             # loss2 = F.cosine_similarity(text_prediction, exp1[sizes.data[0]:]).sum()
#             if CUDA:
#                 idx = idx.cuda()
#
#
#
#             loss3 = F.pairwise_distance(text_prediction, img_feat[idx]).sum()
#             # loss3 = F.cosine_similarity(text_prediction, img_feat[torch.cuda.LongTensor(idx)]).sum()
#             # print('loop', loss.data[0], name.data[0], loss2.data[0], loss3.data[0], time.time() - startl)
#             # print('loop', loss[0], name[0], loss[0], loss3.data[0], time.time() - startl)
#             train_loss += loss3.data[0]  # loss.data[0]
#             loss3.backward()
#             optimizer.step()
#
#             # del loss3, text, img_feat, target, sizes, text_prediction, idx
#             print('batch time', time.time() - startb)
#             # if e < 5:
#             #     for param_group in optimizer.param_groups:
#             #         param_group['lr'] += 0.000004
#             #         print(param_group['lr'])
#             # param_group['lr'] *= lr_decay
#
#         # losslist.append(train_loss)
#         print(train_loss)
#         print(time.time() - start)