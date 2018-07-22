# coding: utf-8
from keras.preprocessing.text import Tokenizer

samples = ['the cat sat on the mat.', 'the dog ate my homework.']

# 出現頻度がもっとも高い1000個の単語だけを処理するトークナイザー
tokenizer = Tokenizer(num_words=1000)

# 単語のインデックスを構築
tokenizer.fit_on_texts(samples)

# 文字列を整数のインデックスのリストに変換
#sequences = tokenizer.texts_to_sequences(samples)

#print(sequences)
# [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

#one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

#print(one_hot_results)
# [[0. 1. 1. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]


# 計算された単語のインデックスを復元
word_index = tokenizer.word_index

print(word_index)
# {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}
