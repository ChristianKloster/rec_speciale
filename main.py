import numpy as np
from matrix_factorization import matrix_factorization as mf
import pandas as pd
import tensorflow as tf
from LRMF import utils

if __name__ == '__main__':

    train = pd.read_csv('EATNN/data/ciao_from_them/extreme_train.csv')
    train.to_csv('EATNN/data/ciao_from_them/extreme_train_without_count.csv', columns=['uid', 'iid'], index=False)
    test = pd.read_csv('EATNN/data/ciao_from_them/extreme_test.csv')
    test.to_csv('EATNN/data/ciao_from_them/extreme_test_without_count.csv', columns=['uid', 'iid'], index=False)