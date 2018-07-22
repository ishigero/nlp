# coding: utf-8
import keras
import keras.callbacks
from keras import preprocessing
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# 特徴量として考慮する単語数
max_features = 10000

# max_len数の単語を残してテキストをカット
max_len = 30

# データを整数のリストとして読み込む
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 変換
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

tb_cb = keras.callbacks.TensorBoard(log_dir='./tflogs/', histogram_freq=1)
cbks = [tb_cb]
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=cbks)



