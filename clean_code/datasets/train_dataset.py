import torch

from datasets.generic_dataset import GenericDataSet
from utilities.data_helpers import pad_text


class QuestionsDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file, histogram_pickle_file, stem=True, stopwords=False, stop_vocab=None, normalize=True,
                 debug=False, augment_binary=True, remove_nonbinary=True, include_captions=False, concat=False):
        self.augment_binary = augment_binary
        self.remove_nonbinary = remove_nonbinary
        self.include_captions = include_captions
        self.concat = concat
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, histogram_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        text_int, target_int = self.convert_question_to_int(row.dialog, stem, stopwords, stop_vocab)
        if self.include_captions:
            caption_int = self.convert_caption_to_int(row.caption, stem, stopwords, stop_vocab)
            if self.concat:
                text_int.extend(caption_int)
                target_int = [1]
            else:
                text_int.append(caption_int)
                target_int = target_int + [1]

        if len(text_int) > 0:
            if not self.concat:
                text_int = pad_text(text_int)
            img_id = row.target_img_id
            text_tensor = torch.LongTensor(text_int)
            if len(text_tensor.size()) < 2:
                target = torch.FloatTensor([target_int])
                text_size = torch.LongTensor([1])
                text_tensor = text_tensor.view(1, -1)
            else:
                target = torch.FloatTensor(target_int)
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
            if answer == 'nope':
                question = question.replace('nope', 'no')
            is_binary = answer == 'yes' or answer == 'no'
            if is_binary:
                question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                question_int = self.text2int(question_preprocessed)
                if len(question_int) > 20:
                    continue
                self.sentences_histograms[len(question_int)] += 1
                answer_int = 1 if answer == 'yes' else -1
                if self.augment_binary:
                    augmented_question_int = question_int[:-1] + [
                        (self.vocab['no'] if answer == 'yes' else self.vocab['yes'])]
                    # BOOKMARK: orig questions
                    augmented_answer_int = -answer_int
                    if self.concat:
                        questions_int.extend(question_int + augmented_question_int)
                    else:
                        answers_int.extend([answer_int, augmented_answer_int])
                        questions_int.extend([question_int, augmented_question_int])
                    # end of bookmark
                else:
                    if self.concat:
                        questions_int.extend(question_int)
                    else:
                        answers_int.append(answer_int)
                        questions_int.append(question_int)
            else:
                if self.remove_nonbinary:
                    continue
                else:
                    question_preprocessed = self.preprocess_text(question, stem, stopwords, stop_vocab)
                    question_int = self.text2int(question_preprocessed)
                    if len(question_int) > 20:
                        continue
                    self.sentences_histograms[len(question_int)] += 1
                    answer_int = 1
                    if self.concat:
                        questions_int.extend(question_int)
                    else:
                        answers_int.append(answer_int)
                        questions_int.append(question_int)
        if self.concat:
            return questions_int, 1
        else:
            return pad_text(questions_int), answers_int


class CaptionsDataSet(GenericDataSet):
    def __init__(self, json_file, pickle_file, img_feat_file, img_map_file,
                 vocab, vocab_pickle_file,histogram_pickle_file,  stem=True, stopwords=False, stop_vocab=None, normalize=True, debug=False):
        super().__init__(json_file, pickle_file, img_feat_file, img_map_file,
                         vocab, vocab_pickle_file, histogram_pickle_file, stem, stopwords, stop_vocab, normalize, debug)

    def convert_to_int(self, row, stem, stopwords, stop_vocab):
        text = row.caption
        caption_preprocessed = self.preprocess_text(text, stem, stopwords, stop_vocab)
        text_int = self.text2int(caption_preprocessed)
        self.sentences_histograms[len(text_int)] += 1
        target = [1]
        if len(text_int) > 0 and len(text_int) < 20:
            img_id = row.target_img_id
            target = torch.FloatTensor(target)
            text_tensor = torch.LongTensor(text_int)
            text_size = torch.LongTensor([1])
            return (text_tensor, img_id, target, text_size)