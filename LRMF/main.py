import pickle

import networkx as nx
import numpy as np
import DivRank as dr
from LRMF import LRMF
from utils import *

if __name__ == '__main__':
    with open('models/ciao_exp_25-75.pkl', 'rb') as f:
        model: LRMF = pickle.load(f)

    ndcg10, ndcg50, ndcg100, prec10, prec50, prec100, recall10, recall50, recall100 = model.evaluate(model.tree)
    print(f'NDCG@10: {ndcg10}. Prec@10: {prec10}. Recall@10: {recall10}')
    print(f'NDCG@50: {ndcg50}. Prec@50: {prec50}. Recall@50: {recall50}')
    print(f'NDCG@100: {ndcg100}. Prec@100: {prec100}. Recall@100: {recall100}')