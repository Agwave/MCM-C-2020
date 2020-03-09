import re
import os
import csv
import numpy as np

from collections import Counter

def get_spec_word_cmp(tsv_path, spec_star):
    all_word_freq = Counter()
    all_word_cnt = 0
    spec_word_freq = Counter()
    spec_word_cnt = 0

    with open(tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, r in enumerate(reader):
            if i == 0:
                continue
            sentence = (r[12] + ' ' + r[13]).lower()
            sentence = re.sub(r'[^A-Za-z0-9]+', ' ', sentence)
            words = sentence.split()
            all_word_freq.update(words)
            all_word_cnt += len(words)
            if r[7] == spec_star:
                spec_word_freq.update(words)
                spec_word_cnt += len(words)
    low_freq_word = []
    for word in spec_word_freq:
        if all_word_freq[word] < all_word_cnt / 10000:
            low_freq_word.append(word)
    for word in low_freq_word:
        if word in spec_word_freq:
            spec_word_freq.pop(word)

    spec_words, spec_word_cmp = [], []
    for word in spec_word_freq.keys():
        spec_words.append(word)
        spec_word_cmp.append((spec_word_freq[word] * all_word_cnt) / (all_word_freq[word] * spec_word_cnt))
    np_spec_word_cmp = np.array(spec_word_cmp)
    sort_idx = np.argsort(-np_spec_word_cmp).tolist()
    return spec_words, spec_word_cmp, sort_idx

if __name__ == '__main__':
    tsv_name = 'hair_dryer.tsv'
    tsv_dir = '/home/agwave/scoures/美赛相关/2020_Weekend2_Problems/Problem_C_Data/'
    tsv_path = os.path.join(tsv_dir, tsv_name)
    spec_words, spec_word_cmp, sort_idx = get_spec_word_cmp(tsv_path, '5')
    print(spec_words)
    print(spec_word_cmp)
    print(sort_idx)
    print()
    for idx in sort_idx[:20]:
        print(spec_words[idx], spec_word_cmp[idx])