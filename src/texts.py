# coding: utf-8
import os

img_dir = '/app/data/aclImdb'
train_dir = os.path.join(img_dir, 'train')
labels = []
texts = None
# for label_type in ['neg', 'pos']:
#   dir_name = os.path.join(train_dir, label_type)
#   for fname in os.listdir(dir_name):
#     if fname[-4:] == '.txt':
#       f = open(os.path.join(dir_name, fname), encoding='utf_8')
#       texts.append(f.read())
#       f.close()
#       if label_type == 'neg':
#         labels.append(0)
#       else:
#         labels.append(1)


# for label_type in ['neg', 'pos']:
#   dir_name = os.path.join(train_dir, label_type)
#   for fname in os.listdir(dir_name):
#     if fname[-4:] == '.txt':
#       f = open(os.path.join(dir_name, fname), encoding='utf_8')
#       texts.append(f.read())
#       f.close()
#       if label_type == 'neg':
#         labels.append(0)
#       else:
#         labels.append(1)
#f = open(os.path.join(dir_name, txt_file), encoding='utf_8')




for label_type in ['neg', 'pos']:
  dir_name = os.path.join(train_dir, label_type)
  text_files = list(map(lambda txt_file: txt_file, filter(lambda f_name: f_name[-4:] == '.txt', os.listdir(dir_name))))
  open_files = list(map(lambda x: open(os.path.join(dir_name, x), encoding='utf_8'), text_files))
  texts = list(map(lambda txt_file: txt_file.read(), open_files))
  lambda o_file: o_file.close(), open_files
label_val = 0 if label_type == 'neg' else 1
labels.append(label_val)

# ジェネレータにする