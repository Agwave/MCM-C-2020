import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from model import BiLSTM
from util import prepare_sequence
from dataset import load_train_data

TRAIN_TAG_PATH = 'tags/train_pacifier.json'
TEST_TAG_PATH = 'tags/test_pacifier.json'
TRAIN_CORPUS_PATH = 'txt_data/train_pacifier.txt'
TEST_CORPUS_PATH = 'txt_data/test_pacifier.txt'
MODEL_NAME = 'train_pacifier_5epoch.pth'
BEST_NAME = 'best_train_pacifier.pth'
EMBEDDING_DIM= 128
HIDDEN_DIM = 128
TRAIN_EPOCH = 5
with open('word_to_ix/train_pacifier_word_to_ix.json', 'r') as j:
    WORD_TO_IX = json.load(j)

def train():
    logging.basicConfig(level=logging.INFO, filename='log.txt', format='%(message)s')
    tag_path = TRAIN_TAG_PATH
    corpus_path = TRAIN_CORPUS_PATH
    save_model_name = MODEL_NAME
    best_model_name = BEST_NAME
    load_model_path = None
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    train_epoch = TRAIN_EPOCH
    word_to_ix = WORD_TO_IX
    start_epoch = 0
    best_score = 0.
    loss_info, train_avg_info, test_avg_info = [], [], []

    sentences, tags = load_train_data(tag_path, corpus_path)
    tag_to_ix = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    label = torch.tensor([[tag_to_ix[tag]] for tag in tags])

    model = BiLSTM(len(word_to_ix), 5, embedding_dim, hidden_dim, dropout=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    if load_model_path is not None:
        checkpoints = torch.load(load_model_path)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optim_state_dict'])
        start_epoch = checkpoints['epoch']

    start_time = time.time()
    logging.info('----------------------')
    for epoch in range(start_epoch, train_epoch):
        running_loss = 0.0
        for i, sen in enumerate(tqdm(sentences)):
            optimizer.zero_grad()
            input = prepare_sequence(sen, word_to_ix)
            output = model(input)
            loss = criterion(output, label[i])
            running_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }, save_model_name)

        train_avg = eval(TRAIN_TAG_PATH, TRAIN_CORPUS_PATH)
        test_avg = eval(TEST_TAG_PATH, TEST_CORPUS_PATH)
        loss_info.append(running_loss)
        train_avg_info.append(train_avg)
        test_avg_info.append(test_avg)

        logging.info('********')
        logging.info('epoch: {}'.format(epoch+1))
        logging.info('loss: {}'.format(running_loss))
        logging.info('train avg: {}'.format(train_avg))
        logging.info('test avg: {}'.format(test_avg))

        if test_avg > best_score:
            torch.save({
                'model_state_dict': model.state_dict(),
            }, best_model_name)
            best_score = test_avg
            print('save best')

    print('training time:', time.time() - start_time)



def eval(tag_path, corpus_path):
    correct = 0
    total = 0
    acc_list = []
    model_name = MODEL_NAME
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    word_to_ix = WORD_TO_IX

    model = BiLSTM(len(word_to_ix), 5, embedding_dim, hidden_dim)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tag_to_ix = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    sentences, tags = load_train_data(tag_path, corpus_path)
    labels = torch.tensor([[tag_to_ix[tag]] for tag in tags[:]])

    with torch.no_grad():
        for i, sen in enumerate(tqdm(sentences[:])):
            input = prepare_sequence(sen, word_to_ix)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            label = labels[i]
            total += label.size(0)
            correct += (predicted == label).sum().item()
            acc = round(100 * correct / total, 2)
            acc_list.append(acc)
    assert len(acc_list) == len(sentences)
    final_acc = acc
    plt.plot(list(range(len(tags))), acc_list)
    plt.xlabel('pred_num')
    plt.ylabel('accuracy / %')
    plt.show()
    return final_acc

def predict(sentence):
    sentence = sentence.split()
    model_name = BEST_NAME
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    word_to_ix = WORD_TO_IX

    model = BiLSTM(len(word_to_ix), 5, embedding_dim, hidden_dim)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    input = prepare_sequence(sentence, word_to_ix)
    with torch.no_grad():
        output = model(input)
        print(output)
        _, predicted = torch.max(output.data, 1)
        print(predicted)



if __name__ == '__main__':
    train()