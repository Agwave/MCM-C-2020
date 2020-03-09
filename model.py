import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, dropout=0):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tag_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=2, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        out = self.out(lstm_out)
        return torch.mean(out, dim=0)

    def init_hidden(self):
        return (torch.randn(4, 1, self.hidden_dim // 2),
                torch.randn(4, 1, self.hidden_dim // 2))
