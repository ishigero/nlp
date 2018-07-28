# coding: utf-8
import os


def _read_file(dir_name, file_names):
  for file_name in file_names:
    with open(dir_name + '/' + file_name, encoding='utf_8') as f:
      yield f.read()


img_dir = '/app/data/aclImdb'
train_dir = os.path.join(img_dir, 'train')
for label_type in ['neg', 'pos']:
  dir_name = os.path.join(train_dir, label_type)
  file_names = list(map(lambda txt_file: txt_file, filter(lambda f_name: f_name[-4:] == '.txt', os.listdir(dir_name))))  
  if label_type == 'neg':
    g_texts_neg = _read_file(dir_name, file_names)
  else:
    g_texts_pos = _read_file(dir_name, file_names)

texts_neg = list(map(lambda text: text, g_texts_neg))
texts_pos = list(map(lambda text: text, g_texts_pos))
label_neg = list(map(lambda txt: 0, texts_neg))  
label_pos = list(map(lambda txt: 1, texts_pos))

texts_neg.extend(texts_pos)
label_neg.extend(label_pos)
texts = texts_neg
labels = label_neg
