# coding: utf-8
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Classifier:
    
    def __init__(self):
        self.texts, self._labels = self._load_data()
        self.x_train, self.y_train, self.x_val, self.y_val = self.tokenize()


    def tokenize(self):
        max_len = 100
        training_samples = 200
        validation_samples = 10000
        max_words = 10000

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=max_len)

        labels = np.asarray(self._labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        x_train = data[:training_samples]
        y_train = labels[:training_samples]
        x_val = data[training_samples: training_samples + validation_samples]
        y_val = labels[training_samples: training_samples + validation_samples]
        return x_train, y_train, x_val, y_val


    def _load_data(self):
        img_dir = '/app/data/aclImdb'
        train_dir = os.path.join(img_dir, 'train')
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(train_dir, label_type)
            file_names = list(map(lambda txt_file: txt_file, filter(lambda f_name: f_name[-4:] == '.txt', os.listdir(dir_name))))  
            if label_type == 'neg':
                g_texts_neg = self._read_file(dir_name, file_names)
            else:
                g_texts_pos = self._read_file(dir_name, file_names)

        texts_neg = list(map(lambda text: text, g_texts_neg))
        texts_pos = list(map(lambda text: text, g_texts_pos))
        label_neg = list(map(lambda txt: 0, texts_neg))  
        label_pos = list(map(lambda txt: 1, texts_pos))

        texts_neg.extend(texts_pos)
        label_neg.extend(label_pos)
        texts = texts_neg
        labels = label_neg
        return texts, labels


    def _read_file(self, dir_name, file_names):
            for file_name in file_names:
                with open(dir_name + '/' + file_name, encoding='utf_8') as f:
                    yield f.read()