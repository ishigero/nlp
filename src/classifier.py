# coding: utf-8
import os
import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

class Classifier:
    
    def __init__(self):
        self.word_index = None
        self.embedding_dim = 100
        self.max_words = 10000
        self.max_len = 30
        self.texts, self._labels = self._load_data()
        self.x_train, self.y_train, self.x_val, self.y_val = self.tokenize()
        self.model = self.build_model(self.word_embedding(self.word_index))


    def build_model(self, e_matrix):
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_len))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.layers[0].set_weights([e_matrix])
        model.layers[0].trainable = False
        model.summary()
        

    def word_embedding(self, w_index):
        e_i = self.read_embedd_file()
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in w_index.items():
            embedding_vector = e_i.get(word)
            if i < self.max_words:
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix


    def read_embedd_file(self):
        glove_dir = '/app/data/glove.6B'
        embeddings_index = {}
        with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf_8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index


    def tokenize(self):
        max_len = 100
        training_samples = 200
        validation_samples = 10000

        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)

        self.word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))
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