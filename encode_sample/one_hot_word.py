# coding: utf-8
import numpy as np


## サンプルのトークン化
samples = ['the cat sat on the mat.', 'the dog ate my homework.']
token_index = {}
for sample in samples:
  for word in sample.split():
    if word not in token_index:
      token_index[word] = len(token_index) + 1

print(token_index)
# {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat.': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework.': 9}


## サンプルのベクトル化(max_lentth個の単語だけを考慮)
max_length = 10
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1 ))

#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]

for i, sample in enumerate(samples):
  for j, word in list(enumerate(sample.split()))[:max_length]:
    print(word)
    # the
    # cat
    # sat
    # on
    # the .....

    index = token_index.get(word)
    print(index)
    # 1
    # 2
    # 3
    # 4
    # 1 ......
    results[i, j, index] = 1.

print(results)

# [[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]


