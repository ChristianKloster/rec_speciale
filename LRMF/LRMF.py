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
from numpy.linalg import inv
from numpy.linalg import norm
import time
import pickle


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20, candidate_items: set = None, num_candidate_items: int = 200):
        '''
        Skriv lige noget om at vi bruger det her paper som kan findes her
        '''
        self.num_candidate_items = num_candidate_items
        self.num_global_questions = num_global_questions
        self.num_local_questions = num_local_questions
        self.train_data, self.test_data = utils.train_test_split(data)
        self.inner_2raw_uid, self.raw_2inner_uid, self.inner_2raw_iid, self.raw_2inner_iid = utils.build_id_dicts(data)
        self.R = utils.build_interaction_matrix(self.train_data, self.raw_2inner_uid, self.raw_2inner_iid)
        self.test_R = utils.build_interaction_matrix(self.test_data, self.raw_2inner_uid, self.raw_2inner_iid)
        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.R.shape
        self.V = np.random.rand(embedding_size, self.num_items)  # Item representations
        self.alpha = alpha
        self.beta = beta

        if candidate_items is not None:
            self.candidate_items = candidate_items
        else:
            self.candidate_items = self._find_candidate_items()
            with open('data/candidate_ciao_exp_25-75.pkl', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 10):
        users = [self.raw_2inner_uid[uid] for uid in self.train_data['uid'].unique()]
        items = self.candidate_items

        best_tree = None
        best_V = None
        best_loss = sys.maxsize
        best_ndcg = sys.maxsize

        ndcg_list = []
        loss = []
        for epoch in range(maxiters):
            maxvol_representatives, _ = maxvol.maxvol(self.V.T)

            # Building tree with global questions
            tree = self._grow_tree(users, items, 0, maxvol_representatives, [])

            # update tree with local questions and transformation matrix
            self._set_globals_learn_locals_and_build_T(tree, maxvol_representatives)

            # Learn item_profiles
            self.V = self._learn_item_profiles(tree)

            # Computing loss
            epoch_loss = self._compute_loss(tree)
            print(f'Epoch: {epoch} done with loss: {epoch_loss}.\n'
                  f'-' * 20)

            loss.append(epoch_loss)

            ndcg, prec, recall = self.evaluate(tree)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_tree = tree
                best_ndcg = ndcg
                best_V = self.V

            print(f'ndcg@10: {ndcg}, prec@10: {prec}, recall@10: {recall}')
            ndcg_list.append(ndcg)

        print(f'The ndcg at the lowest loss was {best_ndcg}')

        self.tree = best_tree
        self.V = best_V
        # CHANGE MODELNAME
        self.store_model('ciao_exp_25-75.pkl')

    def _find_candidate_items(self):
        # building item-item colike network
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting raw iids based on their divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # taking top num_candidate_items
        raw_candidate_iids = sorted_candidate_items[:self.num_candidate_items]
        # translating raw iids to inner iids
        inner_candidate_iids = [self.raw_2inner_iid[iid] for iid in raw_candidate_iids]
        return set(inner_candidate_iids)

    def _grow_tree(self, users, items: set, depth: int,
                   maxvol_iids: list, global_representatives: list = None,
                   total_loss: float = np.inf):
        '''
        :param users: list of uids
        :param items: list of iids on candidate items
        :param depth: depth of tree
        :param global_representatives: items asked previously (defaults to None)
        :return: Tree
        '''
        # Work on parameters
        best_item, like, dislike = None, None, None

        if depth < self.num_global_questions:
            # finding local representatives based on which items are "active", meaning they are consumed
            local_representatives = self._find_local_representatives(users, maxvol_iids, global_representatives)

            print(f'At depth: {depth}. Time: {time.strftime("%H:%M:%S", time.localtime())}')
            print(f'Finding best candidate item')
            loss = defaultdict(float)
            loss_like = defaultdict(float)
            loss_dislike = defaultdict(float)

            # computes loss with equation 11 for each candidate item
            for item in items:
                like, dislike = self._split_users(users, item)

                loss_like[item] = self._evaluate_eq11(like, global_representatives, local_representatives)
                loss_dislike[item] = self._evaluate_eq11(dislike, global_representatives, local_representatives)
                loss[item] += loss_like[item]
                loss[item] += loss_dislike[item]

            # find item with lowest loss
            best_item = min(loss, key=loss.get)
            print(f'Found best candidate item: {best_item} at {time.strftime("%H:%M:%S", time.localtime())}')

            # Lists are funny
            new_global_representatives = global_representatives.copy()
            new_global_representatives.append(best_item)

            # return the node as is if a split wont reduce loss
            if total_loss <= loss[best_item]:
                print(f'Total loss not decreased.\n'
                      f'Total loss: {total_loss}, loss: {loss[best_item]}')
                return Tree.Node(users, None, None, None)

            # else split group into like and dislike
            print(f'Splitting users')
            U_like, U_dislike = self._split_users(users, best_item)

            if not U_like or not U_dislike:
                return Tree.Node(users, None, None, None)

            # building left side (like) of tree
            like = self._grow_tree(U_like, items - {best_item}, depth + 1, maxvol_iids,
                                   new_global_representatives, loss_like[best_item])

            # building right side (dislike) of tree
            dislike = self._grow_tree(U_dislike, items - {best_item}, depth + 1, maxvol_iids,
                                      new_global_representatives, loss_dislike[best_item])

        return Tree.Node(users, best_item, like, dislike)

    def _evaluate_eq11(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)
        pred = B @ T @ self.V

        Rg = self.R[users]
        return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(T)

    def _build_B(self, users, local_representatives, global_representatives):
        # B = [U1, U2, e]
        U1 = self.R[users, :][:, global_representatives].toarray()
        U2 = self.R[users, :][:, local_representatives].toarray()
        return np.hstack((U1, U2, np.ones(shape=(len(users), 1))))

    def _solve_sylvester(self, B, users):
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.R[users] @ self.V.T @ inv(self.V @ self.V.T))

        return T

    def _split_users(self, users, iid):
        like = [uid for uid in users if self.R[uid, iid] >= 4]  # inner uids of users who like inner iid
        dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _find_local_representatives(self, users, maxvol_representatives, global_representatives):
        active_maxvol_items = [iid for iid in maxvol_representatives
                               if np.sum(self.R[users, :][:, iid]) > 0 and iid not in global_representatives]
        if active_maxvol_items:
            return active_maxvol_items[:self.num_local_questions]
        return list(maxvol_representatives[:self.num_local_questions])

    def _learn_item_profiles(self, tree):
        S = np.zeros(shape=(self.num_users, self.embedding_size))

        for inner_uid in range(self.num_users):
            leaf = Tree.find_user_group(inner_uid, tree)
            B = self._build_B([inner_uid], leaf.local_questions, leaf.global_questions)

            try:
                S[inner_uid] = B @ leaf.transformation
            except ValueError:
                print('Arrhhh shit, here we go again')

        return inv(S.T @ S + self.beta * np.identity(self.embedding_size)) @ S.T @ self.R

    def _compute_loss(self, tree):
        if tree.is_leaf():
            B = self._build_B(tree.users, tree.local_questions, tree.global_questions)
            pred = B @ tree.transformation @ self.V
            Rg = self.R[tree.users]
            return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(tree.transformation)

        else:
            return self._compute_loss(tree.like) + self._compute_loss(tree.dislike)

    def _set_globals_learn_locals_and_build_T(self, tree, maxvol, global_questions: list = None):
        if global_questions == None:
            global_questions = []

        if tree.is_leaf():
            tree.set_globals(global_questions)

            locals = self._find_local_representatives(tree.users, maxvol, global_questions)
            tree.set_locals(locals)

            B = self._build_B(tree.users, locals, global_questions)
            tree.set_transformation(self._solve_sylvester(B, tree.users))

        else:
            global_questions.append(tree.question)
            g_q = global_questions.copy()

            self._set_globals_learn_locals_and_build_T(tree.like, maxvol, g_q)
            self._set_globals_learn_locals_and_build_T(tree.dislike, maxvol, g_q)

    def evaluate(self, tree):
        precision10 = 0
        precision50 = 0
        precision100 = 0
        recall10 = 0
        recall50 = 0
        recall100 = 0
        ndcg10 = 0
        ndcg50 = 0
        ndcg100 = 0
        # convert from object to string to numeric..
        users = np.sort(self.test_data.uid.unique())
        R = pd.pivot_table(data=self.test_data, values='rating', index='uid', columns='iid').fillna(0)

        for raw_uid in tqdm(users, desc='Evaluating...'):
            leaf = Tree.traverse_a_user(user=raw_uid, data=self.test_data, tree=tree)  # replace with a traverse
            U1 = np.zeros(shape=(1, len(leaf.global_questions)))
            for idx, gq in enumerate(leaf.global_questions):
                try:
                    U1[:,idx] = int(R[self.raw_2inner_iid[gq]].loc[raw_uid])
                except KeyError:
                    U1[:,idx] = 0

            U2 = np.zeros(shape=(1, len(leaf.local_questions)))
            for idx, lq in enumerate(leaf.local_questions):
                try:
                    U2[:,idx] = int(R[self.raw_2inner_iid[lq]].loc[raw_uid])
                except KeyError:
                    U2[:,idx] = 0

            B = np.hstack((U1, U2, np.ones(shape=(1, 1))))

            ids_to_remove = list(set(self.train_data.iid).difference(set(self.test_data.iid)))
            inner_ids_to_remove = [self.raw_2inner_iid[id] for id in ids_to_remove]
            test_V = np.delete(self.V, inner_ids_to_remove, axis=1)

            T = leaf.transformation
            pred = B @ T @ test_V
            train_iids = list(self.train_data[self.train_data['uid'] == raw_uid].iid)
            questions = leaf.global_questions + leaf.local_questions

            for q in questions:
                inner_q = self.raw_2inner_iid[q]
                try:
                    pred[inner_q] = -np.inf
                except IndexError:
                    pass

            # find actuals
            actual = R.loc[raw_uid].to_numpy()
            m10 = eval2.Metrics2(pred, np.array([actual]), 10, 'ndcg,precision,recall').calculate()
            ndcg10 += m10['ndcg']
            precision10 += m10['precision']
            recall10 += m10['recall']
            m50 = eval2.Metrics2(pred, np.array([actual]), 50, 'ndcg,precision,recall').calculate()
            ndcg50 += m50['ndcg']
            precision50 += m50['precision']
            recall50 += m50['recall']
            m100 = eval2.Metrics2(pred, np.array([actual]), 100, 'ndcg,precision,recall').calculate()
            ndcg100 += m100['ndcg']
            precision100 += m100['precision']
            recall100 += m100['recall']

        n_test_users = len(self.test_data.uid.unique())
        return (ndcg10 / n_test_users), ndcg50 / n_test_users, ndcg100 / n_test_users,\
               precision10 / n_test_users, precision50 / n_test_users, precision100 / n_test_users,\
               recall10 / n_test_users, recall50 / n_test_users, recall100 / n_test_users
        #return ndcg10 / n_test_users, precision10 / n_test_users, recall10 / n_test_users

    def store_model(self, file):
        DATA_ROOT = 'models'
        with open(os.path.join(DATA_ROOT, file), 'wb') as f:
            pickle.dump(self, f)


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
    data = pd.read_csv('../data/ciao_explicit_preprocessed/new_ratings.csv')
    with open('data/candidate_ciao_exp_25-75.pkl', 'rb') as f:
        candidates = pickle.load(f)
    global_questions = 1
    local_questions = 2
    lrmf = LRMF(data, global_questions, local_questions, candidate_items=candidates)
    lrmf.fit()




