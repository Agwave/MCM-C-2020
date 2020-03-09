import os
import re
import csv
import json
import jieba

from util import get_word_to_ix

SPLIT = 16000

def read_data_from_tsv_to_txt(tsv_path, path, train=False):
    corpus = []
    txt_path = path
    if train == True:
        train_path = 'train_' + txt_path
        test_path = 'test_' + txt_path
        train_corpus, test_corpus = [], []
        split = SPLIT
        with open(tsv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                sentence = (r[12] + ' ' + r[13]).lower()
                sentence = re.sub(r'[^A-Za-z0-9,.!]+', ' ', sentence)
                if i <= split:
                    train_corpus.append(sentence)
                else:
                    test_corpus.append(sentence)
        if os.path.exists(train_path):
            os.remove(train_path)
        with open(train_path, 'w+') as f:
            for sen in train_corpus:
                str_cut = jieba.cut(sen)
                s = ' '.join(str_cut)
                f.write(s)
                f.write('\r\n')
        if os.path.exists(test_path):
            os.remove(test_path)
        with open(test_path, 'w+') as f:
            for sen in test_corpus:
                str_cut = jieba.cut(sen)
                s = ' '.join(str_cut)
                f.write(s)
                f.write('\r\n')
    else:
        with open(tsv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                sentence = (r[12] + ' ' + r[13]).lower()
                sentence = re.sub(r'[^A-Za-z0-9,.!]+', ' ', sentence)
                corpus.append(sentence)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        with open(txt_path, 'w+') as f:
            for sen in corpus:
                str_cut = jieba.cut(sen)
                s = ' '.join(str_cut)
                f.write(s)
                f.write('\r\n')

    print('write word finish.')

def write_tag(tsv_path, path, train=False):
    json_path = path
    if train == True:
        split = SPLIT
        train_path = 'train_' + json_path
        test_path = 'test_' + json_path
        train_tags, test_tags = [], []
        with open(tsv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                if i <= split:
                    train_tags.append(r[7])
                else:
                    test_tags.append(r[7])
        with open(train_path, 'w') as j:
            json.dump(train_tags, j)
        with open(test_path, 'w') as j:
            json.dump(test_tags, j)
    else:
        tags = []
        with open(tsv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, r in enumerate(reader):
                if i == 0:
                    continue
                tags.append(r[7])
        with open(json_path, 'w') as j:
            json.dump(tags, j)

    print('write tag finish')

def load_train_data(tag_path, corpus_path):
    with open(tag_path, 'r') as j:
        tags = json.load(j)
    sentences = []
    with open(corpus_path, 'r') as f:
        for line in f.readlines():
            words = line.split()
            sentences.append(words)
    print(len(sentences))
    print(len(tags))
    assert len(sentences) == len(tags)
    return sentences, tags


if __name__ == '__main__':
    # 写词txt
    tsv_name = 'pacifier.tsv'
    tsv_dir = '/home/agwave/scoures/美赛相关/2020_Weekend2_Problems/Problem_C_Data/'
    tsv_path = os.path.join(tsv_dir, tsv_name)
    # txt_path = 'pacifier.txt'
    # read_data_from_tsv_to_txt(tsv_path, txt_path, train=False)

    # 读取数据
    # tag_path = 'hair_dryer.json'
    # txt_path = 'hair_dryer.txt'
    # sentences, tags =  load_train_data(tag_path, txt_path)
    # print(sentences[100])
    # print(tags[100])
    # print(len(sentences), len(tags))
    # with open('hair_dryer.json', 'w') as j:
    #     json.dump(tags, j)
    # word_to_ix = get_word_to_ix(sentences, min_word_freq=5)
    # print(len(word_to_ix))

    write_tag(tsv_path, 'tags/pacifier.json', train=False)


