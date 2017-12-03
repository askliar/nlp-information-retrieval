import h5py
import json
import numpy as np
import os
import pandas as pd

img_data = './data/img_data'
text_data = './data/text_data'

img_features = np.asarray(h5py.File(os.path.join(img_data, 'IR_image_features.h5'), 'r')['img_features'])

with open(os.path.join(img_data, 'IR_img_features2id.json'), 'r') as f:
     visual_feat_mapping = json.load(f)['IR_imgid2id']

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImageTextDataSet(Dataset):

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.imagetext_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.imagetext_frame)

    def __getitem__(self, idx):
        text = self.imagetext_frame.iloc[idx, 0].split(' \n ')
        img_id = self.imagetext_frame.iloc[idx, 1]
        target = self.imagetext_frame.iloc[idx, 2]
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        sample = {'text': text, 'img_features': img_feat, 'target': target}
        return sample


data_train = ImageTextDataSet(csv_file='./data/train_data_easy_bow.csv')
dataloader_train = DataLoader(data_train, batch_size=4,
                        shuffle=True, num_workers=4)

data_val = ImageTextDataSet(csv_file='./data/val_data_easy_bow.csv')
dataloader_val = DataLoader(data_val, num_workers=4)

data_test = ImageTextDataSet(csv_file='./data/test_data_easy_bow.csv')
dataloader_test = DataLoader(data_test, num_workers=4)


# coding: utf-8

"""
CBOW

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]

def read_dataset(dataloader):
    for batch in dataloader:
        text = batch['text']
        for sentences in text:
            for sentence in sentences:
                resulting_sentence = sentence.lower().strip()
                yield ([w2i[x] for x in resulting_sentence.split(" ")])

# Read in the data
train = list(read_dataset(dataloader_train))
w2i = defaultdict(lambda: UNK, w2i)
val = list(read_dataset(dataloader_val))
test = list(read_dataset(dataloader_test))

nwords = len(w2i)

import torch.nn.functional as F
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, img_feat_size, output_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim + img_feat_size, output_dim)
        self.selu = nn.SELU()
        self.linear2 = nn.Linear(output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_text, input_img_feat, batch_size):
        input_text = input_text.view(batch_size, -1)
        x = self.embeddings(input_text)
        x = torch.sum(x, 1)
        x = torch.cat([x, input_img_feat], 1)
        x = self.linear1(x)
        x = self.selu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    

model = CBOW(nwords, 64, 2048, 256).cuda()
print(model)

optimizer = optim.Adam(model.parameters())

def preprocess_text(text):
        max_len = 0
        resulting_text = []
        for sentences in text:
            resulting_sentences = []
            for sentence in sentences:
                resulting_sentence = sentence.lower().strip().split(" ")
                max_len = max(max_len, len(resulting_sentence))
                resulting_sentences.append([w2i[x] for x in resulting_sentence])
            resulting_text.append(resulting_sentences)
        resulting_text = [[sentence + ([UNK] * (max_len - len(sentence)))
                          for sentence in sentences] for sentences in resulting_text]
        return torch.LongTensor(resulting_text)
    
def train(epoch, dataloader, batch_size = 4, cuda = False):
    model.train()
    train_loss = 0.0
    correct = 0.0
    for batch_idx, sample in enumerate(dataloader):
        text_data = preprocess_text(sample['text'])
        img_feat = sample['img_features'] 
        target = sample['target'].float()
        if cuda:
            text_data, img_feat, target = text_data.cuda(), img_feat.cuda(), target.cuda()
        text_data, img_feat, target = Variable(text_data), Variable(img_feat), Variable(target)
        optimizer.zero_grad()
        output = model(text_data, img_feat, batch_size)
        loss = F.binary_cross_entropy(output, target, size_average=False)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0] # sum up batch loss
        pred = output >= 0.5
        pred = pred.view(-1) == target.byte()
        correct += pred.float().cpu().sum().int().data[0]
    train_loss /= len(dataloader.dataset)
    print('\nTrain epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss, correct, len(dataloader.dataset),
        100.0 * correct / len(dataloader.dataset)))

def test(dataloader, batch_size = 1, cuda = False):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, sample in enumerate(dataloader_test):
        text_data = preprocess_text(sample['text'])
        img_feat = sample['img_features'] 
        target = sample['target'].float()
        if cuda:
            text_data, img_feat, target = text_data.cuda(), img_feat.cuda(), target.cuda()
        text_data, img_feat, target = Variable(text_data, volatile=True), Variable(img_feat, volatile=True), Variable(target)
        output = model(text_data, img_feat, batch_size)
        loss = F.binary_cross_entropy(output, target, size_average=False)
        test_loss += loss.data[0] # sum up batch loss
        pred = output >= 0.5
        pred = pred.view(-1) == target.byte()
        correct += pred.float().cpu().sum().int().data[0]
    test_loss /= len(dataloader_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100.0 * correct / len(dataloader.dataset)))
    return 100.0 * correct / len(dataloader.dataset)

import shutil
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

for epoch in range(1, 100):
    train(epoch, dataloader_train, cuda = True)
    best_prec1 = 0.0
    prec1 = test(dataloader_val, cuda = True)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
