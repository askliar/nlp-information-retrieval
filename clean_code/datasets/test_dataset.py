import torch

from datasets.generic_dataset import GenericDataSet
from utilities.data_helpers import pad_text


class TestDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, histogram_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True,
                 debug=False, augment_binary=False, remove_nonbinary=True, concat=False):
        self.augment_binary = augment_binary
        self.remove_nonbinary = remove_nonbinary
        self.concat = concat
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, histogram_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        questions_int = self.convert_question_to_int(row.dialog, stem, stopwords, stop_vocab)
        caption_int = self.convert_caption_to_int(row.caption, stem, stopwords, stop_vocab)
        if self.concat:
            text_int = questions_int + caption_int
        else:
            text_int = pad_text(questions_int + [caption_int])
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
                self.sentences_histograms[len(question_int)] += 1
                if self.augment_binary:
                    augmented_question_int = question_int[:-1] + [
                        (self.vocab['no'] if answer == 'yes' else self.vocab['yes'])]
                    # BOOKMARK: orig questions
                    if self.concat:
                        questions_int.extend(question_int + augmented_question_int)
                    else:
                        questions_int.extend([question_int, augmented_question_int])
                    # end of bookmark
                else:
                    if self.concat:
                        questions_int.extend(question_int)
                    else:
                        questions_int.append(question_int)
            else:
                if self.remove_nonbinary:
                    continue
                else:
                    question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                    question_int = self.text2int(question_preprocessed)
                    self.sentences_histograms[len(question_int)] += 1
                    if self.concat:
                        questions_int.extend(question_int)
                    else:
                        questions_int.append(question_int)
        if self.concat: #add rnn1
            return questions_int
        else:
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