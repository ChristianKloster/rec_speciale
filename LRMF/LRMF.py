import os
from collections import defaultdict

from tqdm import tqdm
from scipy.linalg import solve_sylvester
import scipy.sparse

import numpy as np

import evaluation.evaluation_v2 as eval2
import sys
import DivRank as dr
import Tree
import pandas as pd
import utils
import maxvol
from maxvol2 import py_rect_maxvol
from numpy.linalg import inv
from numpy.linalg import norm
import time
import pickle


def local_questions_vector(candidates: list, entity_embeddings: np.ndarray, max_length: int):
    questions, _ = py_rect_maxvol(entity_embeddings[list(candidates)], maxK=max_length)
    return questions[:max_length].tolist()


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, use_saved_data = False, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20, candidate_items: set = None, num_candidate_items: int = 200):
        '''
        Skriv lige noget om at vi bruger det her paper som kan findes her
        '''
        self.num_candidate_items = num_candidate_items
        self.num_global_questions = num_global_questions
        self.num_local_questions = num_local_questions

        if use_saved_data:
            self.train_data = pd.read_csv('LRMF_data/ciao_explicit/exp_ciao_extreme_train_70_30.csv', sep=',')
            self.test_data = pd.read_csv('LRMF_data/ciao_explicit/exp_ciao_extreme_test_70_30.csv', sep=',')
        else:
            print('No data')

        self.R = pd.pivot_table(self.train_data.astype(str).astype(int), values='rating', index='uid', columns='iid').fillna(0)
        self.test_R = pd.pivot_table(self.test_data.astype(str).astype(int), values='rating', index='uid', columns='iid').fillna(0)

        self.ratings = self.R.to_numpy()
        self.test_ratings = self.test_R.to_numpy()
        self.train_iids = list(self.R.columns)
        self.test_iids = list(self.test_R.columns)

        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.ratings.shape
        self.num_test_users = self.test_ratings.shape[0]

        self.candidate_raw_2_inner = dict(zip(self.train_data.iid.unique(), list(range(self.num_items))))
        self.candidate_inner_2_raw = dict(zip(list(range(self.num_items)), self.train_data.iid.unique()))

        self.V = np.random.rand(embedding_size, self.num_items)  # Item representations
        self.alpha = alpha
        self.beta = beta

        if candidate_items is not None:
            self.candidate_items = candidate_items
        else:
            self.candidate_items = self._find_candidate_items()
            with open('LRMF_data/ciao_implicit/ciao_implicit_candidate_items.txt', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 10):
        users = [u for u in range(self.num_users)]
        best_tree = None
        best_V = None
        best_loss = sys.maxsize
        best_ndcg = sys.maxsize
        best_ten = None
        best_fifty = None
        best_hundred = None

        ndcg_list = []
        loss = []
        for epoch in range(maxiters):
            maxvol_representatives, _ = maxvol.maxvol(self.V.T)

            tree = self._grow_tree(users, set(self.candidate_items), 0, maxvol_representatives, [])

            self.V = self._learn_item_profiles(tree)

            epoch_loss = self._compute_loss(tree)

            loss.append(epoch_loss)
            #ten, fifty, hundred = self.evaluate(tree)
            #ten = self.evaluate(tree)
            print('bp')

            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_tree = tree
               # best_ten = ten
               # best_fifty = fifty
               # best_hundred = hundred
                best_V = self.V

        self.tree = best_tree
        self.V = best_V
        self.best_ten = best_ten
        self.best_fifty = best_fifty
        self.best_hundred = best_hundred
        self.store_model("LRMF_models/ciao_explicit/extreme_ciao_best_lrmf_model.pkl")

    def _find_candidate_items(self):
        # building item-item colike network
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting candidates based on divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # translate to inner ids so they match with self.V
        sorted_and_translated_candidate_items = [self.candidate_raw_2_inner[cand] for cand in sorted_candidate_items]
        # return number of candidate items wanted (200)
        return sorted_and_translated_candidate_items[:self.num_candidate_items]

    def _grow_tree(self, users, items: set, depth: int,
                   maxvol_iids: list, global_representatives: list):
        '''
        :param users: list of uids
        :param items: list of iids on candidate items
        :param depth: depth of tree
        :param global_representatives: items asked previously (defaults to None)
        :return: Tree
        '''
        current_node = Tree.Node(users, None, None, None, None)
        best_question, like, dislike = None, None, None

        if depth == self.num_global_questions:
            local_representatives = local_questions_vector(self.candidate_items, self.V.T, self.num_local_questions)
            current_node.set_locals(local_representatives)
            current_node.raw_locals = [self.candidate_inner_2_raw[lq] for lq in local_representatives]

            current_node.set_globals(global_representatives)
            current_node.raw_globals = ([self.candidate_inner_2_raw[gq] for gq in global_representatives])

            current_node.question = current_node.global_questions[-1:]
            current_node.raw_question = self.candidate_inner_2_raw[current_node.global_questions[-1]]

            B = self._build_B(users, local_representatives, global_representatives)
            current_node.set_transformation(self._solve_sylvester(B, users))

        if depth < self.num_global_questions:
            # computes loss with equation 11 for each candidate item
            min_loss, best_question, best_locals = np.inf, None, []
            for item in tqdm(self.candidate_items, desc=f'[Selecting question at depth {depth} ]'):
                like, dislike = self._split_users(users, item)
                loss = 0

                for group in [like, dislike]:
                    rest_candidates = [i for i in self.candidate_items if not i == item]  # Inner candidates
                    local_questions = local_questions_vector(rest_candidates, self.V.T, self.num_local_questions)
                    loss += self._group_loss(group, global_representatives, local_questions)

                if loss < min_loss:
                    min_loss = loss
                    best_question = item
                    best_locals = local_questions

            g_r = list(global_representatives).copy()
            g_r.append(best_question)

            current_node.question = best_question
            current_node.raw_question = self.candidate_inner_2_raw[best_question]

            current_node.set_locals(best_locals)
            current_node.raw_locals = ([self.candidate_inner_2_raw[iid] for iid in best_locals])

            U_like, U_dislike = self._split_users(users, best_question)

            if not U_like or not U_dislike:
                current_node.child = self._grow_tree(users, items - {best_question}, depth + 1, maxvol_iids, g_r)

            else:
                current_node.like = self._grow_tree(U_like, items - {best_question}, depth + 1, maxvol_iids, g_r)
                current_node.dislike = self._grow_tree(U_dislike, items - {best_question}, depth + 1, maxvol_iids, g_r)
        return current_node

    def _group_loss(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)

        Rg = self.ratings[users]
        pred = B @ T @ self.V
        loss = ((Rg - pred) ** 2).sum()
        regularisation = T.sum() ** 2

        return loss + regularisation

    def _build_B(self, users, local_representatives, global_representatives):
        # B = [U1, U2, e]
        U1 = self.ratings[users, :][:, global_representatives]
        U2 = self.ratings[users, :][:, local_representatives]
        return np.hstack((U1, U2, np.ones(shape=(len(users), 1))))

    def _solve_sylvester(self, B, users):
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.ratings[users] @ self.V.T @ inv(self.V @ self.V.T))

        return T

    def _split_users(self, users, iid):
        if implicit:
            like = [uid for uid in users if self.ratings[uid, iid] >= 1]
            dislike = list(set(users) - set(like))
        else:
            like = [uid for uid in users if self.ratings[uid, iid] >= 4] # inner uids of users who like inner iid
            dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _learn_item_profiles(self, tree):
        S = np.zeros(shape=(self.num_users, self.embedding_size))

        for user in tqdm(range(self.num_users), desc="[Optimizing entity embeddings...]"):
            leaf = Tree.traverse_a_user(user, self.ratings[user], tree)
            B = self._build_B([user], leaf.local_questions, leaf.global_questions)

            try:
                S[user] = B @ leaf.transformation
            except ValueError:
                print('Arrhhh shit, here we go again')

        return (inv(S.T @ S + self.beta * np.identity(self.embedding_size)) @ S.T @ self.ratings)

    def _compute_loss(self, tree):
        if tree.is_leaf():
            B = self._build_B(tree.users, tree.local_questions, tree.global_questions)
            pred = B @ tree.transformation @ self.V
            Rg = self.ratings[tree.users]
            return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(tree.transformation)

        else:
            if tree.child == None:
                return self._compute_loss(tree.like) + self._compute_loss(tree.dislike)
            else:
                return self._compute_loss(tree.child)


    def evaluate(self, tree):
        item_profiles = pd.DataFrame(self.V.copy())

        actual = self.test_R.copy().to_numpy()

        users = list(self.test_R.index)
        user_profiles = pd.DataFrame(data=0, index=users, columns=range(20))

        u_a_empty = []


        for user in users:
            try:
                answers = self.R.loc[user]
            except KeyError:
                answers = self.test_R.loc[user]

            user_profiles.loc[user] = self.interview_new_user(answers, u_a_empty, tree)

        pred = user_profiles.to_numpy() @ item_profiles

        pred = pred.to_numpy()
        actual = (actual != 0).astype(int)

        print(f'----- Computing metrics for k10')
        m10 = eval2.Metrics2(pred, actual, 10, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k50')
        m50 = eval2.Metrics2(pred, actual, 50, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k100')
        m100 = eval2.Metrics2(pred, actual, 100, 'ndcg, precision, recall').calculate()

        return m10, m50, m100

    def interview_new_user(self, actual_answers, user_answers, tree):
        if tree.is_leaf():
            # Have we asked all our local questions?
            if len(user_answers) < len(tree.global_questions) + len(tree.local_questions):
                for local_question in tree.local_questions:
                    # First try to exhaust the available answers
                    try:
                        answer = actual_answers[local_question]
                        u_a = user_answers.copy()
                        u_a.append(answer)
                        return self.interview_new_user(actual_answers, u_a, tree)

                    # If we cannot get an answer from the arguments, return the question
                    except IndexError:
                        return local_question

            # If we have asked all of our questions, return the transformed user vector
            else:
                user_vector = [a for a in user_answers]
                user_vector.append(1)  # Add bias
                return np.array(user_vector) @ tree.transformation

        # find answer to global question
        try:
            answer = actual_answers[tree.question]
        except KeyError:
            print(f'item {tree.raw_question} was not found in test, setting answer to 0')
            answer = 0

        u_a = user_answers.copy()
        u_a.append(answer)

        if implicit:
            if tree.child == None:
                if answer >= 1:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                else:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

        else:
            if tree.child == None:
                if answer >= 4:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                else:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

    def store_model(self, file):
        DATA_ROOT = ''
        with open(os.path.join(DATA_ROOT, file), 'wb') as f:
            pickle.dump(self, f)

    def test_cold(self, tree, k=10):
        ndcg_0 = []
        ndcg_1 = []
        ndcg_2 = []
        ndcg_3 = []
        ndcg_4 = []
        ndcg_5 = []
        ndcg_6 = []
        ndcg_7 = []
        ndcg_8 = []
        ndcg_9 = []
        ndcg_10 = []

        users = [u for u in self.test_data['uid'].unique()]
        for user in tqdm(users):
            try:
                train_ratings = self.R.loc[user]
                train_length = train_ratings.astype(bool).sum(axis=0)
            except KeyError:
                train_length = 0

            if (train_length) > 10:
                continue

            u_a_empty = []
            user_profiles = self.interview_new_user(self.test_R.loc[user], u_a_empty, tree)

            actuals = self.test_ratings[user]
            preds = user_profiles @ self.V.copy()
            top_k_items = preds.argsort()[-k:][::-1]

            tp = 1. / np.log2(np.arange(2, k + 2))
            dcg_relevances = [1 if actuals[p] > 0 else 0 for p in top_k_items]

            DCG = np.sum(dcg_relevances * tp)
            IDCG = tp[:min(len(actuals), k)].sum()

            if train_length == 10: ndcg_10.append(DCG/IDCG)
            if train_length == 9: ndcg_9.append(DCG/IDCG)
            if train_length == 8: ndcg_8.append(DCG/IDCG)
            if train_length == 7: ndcg_7.append(DCG/IDCG)
            if train_length == 6: ndcg_6.append(DCG/IDCG)
            if train_length == 5: ndcg_5.append(DCG/IDCG)
            if train_length == 4: ndcg_4.append(DCG/IDCG)
            if train_length == 3: ndcg_3.append(DCG/IDCG)
            if train_length == 2: ndcg_2.append(DCG/IDCG)
            if train_length == 1: ndcg_1.append(DCG/IDCG)
            if train_length == 0: ndcg_0.append(DCG/IDCG)

        return np.mean(ndcg_0), np.mean(ndcg_1), np.mean(ndcg_2), np.mean(ndcg_3), \
               np.mean(ndcg_4), np.mean(ndcg_5), np.mean(ndcg_6), np.mean(ndcg_7), \
               np.mean(ndcg_8), np.mean(ndcg_9), np.mean(ndcg_10)



def test_tree(users, items, depth):
    like, dislike, best_item = None, None, None

    if depth < 2 or not users:
        best_item = min(items)
        if 3 in users:
            best_item = 3
        split_idx = round(len(users) / 2)

        U_like = users[:split_idx]
        U_dislike = users[split_idx:]

        like = test_tree(U_like, items - {best_item}, depth + 1)
        dislike = test_tree(U_dislike, items - {best_item}, depth + 1)

    return Tree.Node(users, best_item, like, dislike)


if __name__ == '__main__':
    data = pd.read_csv('data/ciao/ratings.csv')

    implicit = True  # Set to true for consumption instead of explicit results
    global_questions = 3
    local_questions = 2

    with open('../LRMF/data/candidate_ciao_70_30.pkl', 'rb') as f:
        candidates = pickle.load(f)

    #with open("LRMF_models/ciao_explicit/LRMF_best_model_normal_cold_start_ciao_exp_consumption.pkl", 'rb') as f:
    #    lrmf = pickle.load(f)

   # zero, one, two, three, four, five, sixs, seven, eight, nine, ten = lrmf.test_cold(lrmf.tree)
   # print('bp')
  #  t, f, h = lrmf.evaluate(lrmf.tree)
   # print('bp')

    lrmf = LRMF(data, global_questions, local_questions, use_saved_data=True, candidate_items=candidates)
    #lrmf = LRMF(data, global_questions, local_questions, use_saved_data=True)
    lrmf.fit()

 #   load data

