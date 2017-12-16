import numpy as np
import matplotlib.pyplot as plt
import torch


def pad_text(text):
    max_len = 0
    if type(text) == int:
        text = [text]
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

def plot_histogram(histogram_dict):
    labels, values = zip(*histogram_dict.items())

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()