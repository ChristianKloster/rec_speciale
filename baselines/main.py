import pickle
import pandas as pd

from baselines import BPR, BPR_utils, SBPR_theano
from baselines.QSBPR import QSBPR
from LRMF import *


def run_BPR():
    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/EATNN_ciao/training_ciao_implicit_25_75_without_ratings.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/EATNN_ciao/testing_ciao_implicit_25_75_without_ratings.csv',
                                                                     uid_raw_to_inner, iid_raw_to_inner)

    model = BPR.BPR(32, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    model.train(train_data)

    for k in [10, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')

def run_SBPR():
    # loading interaction data
    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(
        '../data/EATNN_ciao/training_ciao_implicit_25_75_without_ratings.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(
        '../data/EATNN_ciao/testing_ciao_implicit_25_75_without_ratings.csv',
        uid_raw_to_inner, iid_raw_to_inner)
    # loading social data
    social_data = pd.read_csv('../data/EATNN_ciao/ciao_implicit_trust.csv')

    model = SBPR_theano.SBPR(10, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    model.train(train_data, social_data)

    for k in [10, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')


def run_QSBPR():
    # loading interaction data
    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(
        '../data/EATNN_ciao/training_ciao_implicit_25_75_without_ratings.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(
        '../data/EATNN_ciao/testing_ciao_implicit_25_75_without_ratings.csv',
        uid_raw_to_inner, iid_raw_to_inner)
    # loading social data
    social_data = pd.read_csv('../data/EATNN_ciao/ciao_implicit_trust.csv')
    # tree
    with open('../LRMF/models/ciao_implicit_best_model.pkl', 'rb') as f:
        lrmf = pickle.load(f)
    tree = lrmf.tree

    model = QSBPR(10, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    model.train(train_data, social_data, tree)

    for k in [10, 50, 100]:
         ndcg, prec, recall = model.test(test_data, k=k)
         print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')


if __name__ == '__main__':
    #run_BPR()
    #run_SBPR()
    run_QSBPR()
