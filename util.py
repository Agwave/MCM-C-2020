import torch
from collections import Counter

def get_word_to_ix(sentences, min_word_freq=5):
    word_freq = Counter()
    for sentence in sentences:
        word_freq.update(sentence)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_to_ix = {k: v for v, k in enumerate(words)}
    word_to_ix['<unk>'] = len(word_to_ix)
    return word_to_ix


def prepare_sequence(seq, to_ix):
    unk_ix = to_ix['<unk>']
    idxs = [to_ix.get(w, unk_ix) for w in seq]
    return torch.tensor(idxs)


