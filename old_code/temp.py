from torch.utils.data import Dataset, DataLoader
import torch
import pickle

class A(object):
    def x(self):
        self.y()
    def y(self):
        print('default behavior')


class B(A):
    def y(self):
        print('Child B behavior')

class C(A):
    def y(self):
        print('Child C behavior')

class D(A,B):
    def z(self):
        super(D).y()


class TestDataSet(Dataset):
    def __init__(self, json_file, pickle_file, img_features, visual_feat_mapping,
                 mean, std, vocab, stem=False, stopwords=False, stop_vocab = None, normalize=True, debug=False):
        self.df = pd.read_json(json_file).T.sort_index()

        if debug:
            self.df = self.df.head(5)

        self.img_features = img_features
        self.visual_feat_mapping = visual_feat_mapping
        self.mean = mean
        self.std = std
        self.vocab = vocab
        self.stem = stem
        self.stopwords = stopwords
        if stopwords:
            self.stop_vocab = stop_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption[idx]
        dialog = self.df.dialog[idx]
        imgids = self.df.img_list
        target = self.df.target
        resulting_dialog = self.questions2int(dialog, self.vocab)
        resulting_caption = self.caption2int(caption, self.vocab)
        text_tensor = torch.FloatTensor(self.pad_text(resulting_caption + resulting_dialog))
        target = torch.FloatTensor(target)
        imgs = torch.FloatTensor(self.imgids2imgs(imgids))
        return (text_tensor, imgs, target)

    def text2int(self, text, vocab):
        return [vocab[word] for word in text]

    def caption2int(self, caption, vocab):
        processed_caption = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                   for word in word_tokenize(caption.lower())
                                   if not self.stopwords or word not in self.stop_vocab]
                                  )
        processed_caption = self.text2int(processed_caption, vocab)
        return processed_caption

    def questions2int(self, questions, vocab):
        max_len = 0
        questions = [sentence for sentences in questions for sentence in questions]
        questions_int = []
        for question in questions:
            answer = question.split('?')[1].strip().lower()
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                processed_question = tuple([(lambda x: stemmer.stem(x) if self.stem else x)(word)
                                            for word in word_tokenize(question.lower())
                                            if not self.stopwords or word not in self.stop_vocab]
                                           )
                question_int = self.text2int(processed_question, vocab)
                questions_int.append(question_int)
        return self.pad_text(questions_int)

    def pad_text(self, text):
        max_len = 0
        for sentence in text:
            max_len = max(max_len, len(sentence))
        resulting_questions = [sentence + ([0] * (max_len - len(sentence))) for sentence in text]
        return resulting_questions

    def imgids2imgs(self, img_id_arr):
        imgs = []
        for img_id in img_id_arr:
            h5id = self.visual_feat_mapping[str(img_id)]
            img = self.img_features[h5id]
            img = (img - self.mean) / self.std
            imgs.append(img)
        return imgs

complexity = 'easy'


tempdataset = TestDataSet(json_file=json_val_file,
                          pickle_file=pickle_val_file,
                          img_features=)
#
# tempdataset = TrainDataSet()
# tempdataloader = DataLoader(tempdataset, batch_size=2, shuffle=False, num_workers=4)
#
# for batch in tempdataloader:
#     print(batch[0])
import pandas as pd
import os

text_data = './data/text_data'
complexity = 'easy'



df = preprocess_test_data(complexity)