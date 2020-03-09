# import torch
# A = torch.tensor([[1], [2], [3]])
# print(A[1])
# import re
# s = 'ste...fwef.fwef'
# a = re.sub(r'\.{2,3,4,5,6,7}', '', s)
# print(a)
# from collections import Counter
# from util import get_word_to_ix

# def get_word_to_count(corpus_path):
#     sentences = []
#     with open(corpus_path, 'r') as f:
#         for line in f.readlines():
#             words = line.split()
#             sentences.append(words)
#
#     word_freq = Counter()
#     for sen in sentences:
#         word_freq.update(sen)
#     return word_freq
#
# if __name__ == '__main__':
#     txt_path = 'hair_dryer.txt'
#     word_freq = get_word_to_count(txt_path)
#     print(word_freq)
#     print(len(word_freq))


# import matplotlib.pyplot as plt
# loss_info = [10354.068624436855, 7647.285599470139, 6514.505899012089, 5552.083805263042, 4596.30319583416]
# plt.plot(list(range(1, 6)), loss_info)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('BiLSTM Model')
# plt.show()

# avg = 89.161
# print('Accuracy: %d %%' %avg)

# import json
# from util import get_word_to_ix
# from dataset import load_train_data
#
# sentences, _ = load_train_data('train_hair_dryer.json', 'train_hair_dryer.txt')
# word_to_ix = get_word_to_ix(sentences, min_word_freq=5)
# print(len(word_to_ix))
# with open('train_hair_dryer_word_to_ix.json', 'w') as j:
#     json.dump(word_to_ix, j)
#
import torch

a = torch.tensor([2])
print(a.item())