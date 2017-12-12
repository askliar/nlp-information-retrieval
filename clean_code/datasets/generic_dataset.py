import h5py
import torch
from textblob.tokenizers import word_tokenize
from torch.utils.data import Dataset

from clean_code.utilities.nltk_helpers import stemmer


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
