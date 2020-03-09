import csv
import os
import re
import json
import torch

from train import EMBEDDING_DIM, HIDDEN_DIM
from model import BiLSTM
from util import prepare_sequence

# hair_dryer    423960      587
# microwave     109226352   394
# pacifiler     723849      833

def get_idx_by_year_month(year, month):
    start_year = 2014
    start_month = 1
    return 12 * (year - start_year) + (month - start_month)


def get_time_to_score(tsv_path, thing, model_path):
    time_to_count = {}
    time_to_scoresum = {}
    if thing == 'hair_dryer':
        id = '732252283'
    elif thing == 'microwave':
        id = '423421857'
    else:
        id = '246038397'

    with open('train_'+thing+'_word_to_ix.json', 'r') as j:
        word_to_ix = json.load(j)
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    model = BiLSTM(len(word_to_ix), 5, embedding_dim, hidden_dim)
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()

    with open(tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, r in enumerate(reader):
            if i == 0 or r[4] != id:
                continue
            month, _, year = r[14].split('/')
            if year not in  {'2014', '2015'}:
                continue
            time = get_idx_by_year_month(int(year), int(month))
            if time < 8:
                continue
            sen = (r[12] + ' ' + r[13]).lower()
            sen = re.sub(r'[^A-Za-z0-9,.!]+', ' ', sen)
            input = prepare_sequence(sen.split(), word_to_ix)
            with torch.no_grad():
                output = model(input)
                _, predicted = torch.max(output.data, 1)
            pred_score = predicted.item()
            if time not in time_to_count:
                time_to_count[time] = 0
                time_to_scoresum[time] = 0.
            time_to_count[time] += 1
            time_to_scoresum[time] += pred_score
    time_to_scoremean = {}
    for time in time_to_count.keys():
        time_to_scoremean[time] = time_to_scoresum[time] / time_to_count[time]
    print(time_to_count)
    return time_to_scoremean

# def compute_score_by_time(time_to_ids):
#     for

if __name__ == '__main__':
    tsv_name = 'pacifier.tsv'
    tsv_dir = '/home/agwave/scoures/美赛相关/2020_Weekend2_Problems/Problem_C_Data/'
    tsv_path = os.path.join(tsv_dir, tsv_name)
    thing = 'pacifier'
    model_path = 'model/best_train_pacifier.pth'
    time_to_score = get_time_to_score(tsv_path, thing, model_path)
    print(time_to_score)