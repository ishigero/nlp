import numpy as np
from texts import texts
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_len = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
print(texts[0])
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(word_index)