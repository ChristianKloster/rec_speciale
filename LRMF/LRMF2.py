from typing import List, Tuple, Union, Dict

import numpy as np
import pickle
import utils
import pandas as pd
import evaluation.evaluation_v2 as eval
#from loguru import logger
from scipy.linalg import solve_sylvester, inv, LinAlgError
from maxvol2 import py_rect_maxvol
from tqdm import tqdm





LIKE = 5
DISLIKE = 1


def except_item(lst, item):
    return [i for i in lst if not i == item]


def group_representation(users: List[int],
                         l1_questions: List[int], l2_questions: List[int],
                         ratings: np.ndarray) -> np.ndarray:
    ratings_submatrix = ratings[users][:, l1_questions + l2_questions]
    with_bias = np.hstack((ratings_submatrix, np.ones((len(users), 1))))
    return with_bias


def split_users(users: List[int], entity: int, ratings: np.ndarray) -> Tuple[List[int], List[int]]:
    user_ratings = ratings[users, entity]
    likes, = np.where(user_ratings > 4)
    dislikes, = np.where(user_ratings <= 4)
    return likes, dislikes


def global_questions_vector(questions: List[int], max_length: int) -> List[int]:
    padding = [-1 for _ in range(max_length - len(questions))]
    return questions + padding


def local_questions_vector(candidates: List[int], entity_embeddings: np.ndarray, max_length: int) -> List[int]:
    questions, _ = py_rect_maxvol(entity_embeddings[candidates], maxK=max_length)
    return questions[:max_length].tolist()


def group_loss(users: List[int], global_questions: List[int], local_questions: List[int],
               entity_embeddings: np.ndarray, ratings: np.ndarray) -> float:
    _B = group_representation(users, global_questions, local_questions, ratings)
    _T = transformation(users, _B, entity_embeddings, ratings)

    group_ratings = ratings[users]
    predicted_ratings = _B @ _T @ entity_embeddings.T
    loss = ((group_ratings - predicted_ratings) ** 2).sum()
    regularisation = _T.sum() ** 2

    return loss + regularisation


def transformation(users: List[int],
                   representation: np.ndarray, entity_embeddings: np.ndarray,
                   ratings: np.ndarray, alpha=1.0) -> np.ndarray:
    try:
        _A = representation.T @ representation
        _B = alpha * inv(entity_embeddings.T @ entity_embeddings)
        _Q = representation.T @ ratings[users] @ entity_embeddings @ inv(entity_embeddings.T @ entity_embeddings)
        return solve_sylvester(_A, _B, _Q)
    except LinAlgError as err:
        return np.zeros((representation.shape[1], entity_embeddings.shape[1]))


def optimise_entity_embeddings(ratings: np.ndarray, tree, k: int, kk: int, regularisation: float) -> np.ndarray:
    n_users, n_entities = ratings.shape
    _S = np.zeros((n_users, kk))

    for u in tqdm(range(n_users), desc="[Optimizing entity embeddings...]"):
        _S[u] = tree.interview_existing_user(u)

    try:
        _A = inv(_S.T @ _S) + np.eye(kk) * regularisation
        _B = _S.T @ ratings
        return (_A @ _B).T
    except LinAlgError as err:
        return np.zeros((n_entities, kk))


class LRMF:
    def __init__(self, n_users: int, n_entities: int, l1: int, l2: int, kk: int, regularisation: float):
        """
        Instantiates an LRMF model for conducting interviews and making
        recommendations. The model conducts an interview of total length L = l1 + l2.

        :param n_users: The number of users.
        :param n_entities: The number of entities.
        :param l1: The number of global questions to be used for group division.
        :param l2: The number of local questions to be asked in every group.
        :param kk: The number of latent factors for entity embeddings.
                   NOTE: Due to a seemingly self-imposed restriction from the paper, the number
                   of latent factors used to represent entities must be exactly l2, and cannot be kk.
                   We have emailed the authors requesting an explanation and are awaiting a response.
        :param regularisation: Control parameter for l2-norm regularisation.
        """
        self.n_users = n_users
        self.n_entities = n_entities
        self.l1 = l1
        self.l2 = l2
        self.kk = kk  # See the note
        self.regularisation = regularisation

        self.interview_length: int = self.l1 + self.l2
        self.k: int = self.interview_length + 1

        self.ratings: np.ndarray = np.zeros((self.n_users, self.n_entities))
        self.entity_embeddings: np.ndarray = np.random.rand(self.n_entities, self.kk)

        self.T = Tree(l1_questions=[], depth=0, max_depth=self.l1, lrmf=self)

    def fit(self, ratings: np.ndarray, candidates: List[int]):
        self.ratings = ratings

        all_users = [u for u in range(self.n_users)]

        self.T.grow(all_users, candidates)
        self.entity_embeddings = optimise_entity_embeddings(ratings, self.T, self.k, self.kk, self.regularisation)

    def validate(self, user: int, to_validate: List[int]) -> Dict[int, float]:
        user_vector = self.T.interview_existing_user(user)
        similarities = user_vector @ self.entity_embeddings[to_validate].T
        return {e: s for e, s in zip(to_validate, similarities)}

    def interview(self, answers: Dict[int, int]) -> int:
        return self.T.interview_new_user(answers, {})

    def rank(self, items: List[int], answers: Dict[int, int]):
        user_vector = self.T.interview_new_user(answers, {})
        similarities = user_vector @ self.entity_embeddings[items].T
        return {e: s for e, s in zip(items, similarities)}

    def _val(self, users):
        predictions = []
        for u_idx, user in users.items():
            prediction = self.validate(u_idx, user.validation.to_list())
            predictions.append((user.validation, prediction))


class Tree:
    def __init__(self, l1_questions: List[int], depth: int, max_depth: int, lrmf: LRMF):
        self.depth = depth
        self.max_depth = max_depth
        self.l1_questions = l1_questions
        self.l2_questions: List[int] = []
        self.lrmf = lrmf

        self.users: List[int] = []
        self.transformation: Union[np.ndarray, None] = None

        self.question: Union[int, None] = None

        self.children: Union[Dict[int, Tree], None] = None

    def is_leaf(self):
        return self.depth == self.max_depth

    def grow(self, users: List[int], candidates: List[int]):
        self.users = users
        self.l2_questions = local_questions_vector(
            list(candidates), self.lrmf.entity_embeddings, self.lrmf.l2)

        if self.is_leaf():
            self.transformation = transformation(
                self.users, group_representation(self.users, self.l1_questions, self.l2_questions, self.lrmf.ratings),
                self.lrmf.entity_embeddings, self.lrmf.ratings)
            return

        min_loss, best_question = np.inf, None
        for candidate in tqdm(candidates, desc=f'[Selecting question at depth {self.depth} ]'):
            likes, dislikes = split_users(users, candidate, self.lrmf.ratings)

            loss = 0
            for group in [likes, dislikes]:
                rest_candidates = except_item(candidates, candidate)
                global_questions = global_questions_vector(self.l1_questions, self.lrmf.l1)
                local_questions = local_questions_vector(rest_candidates, self.lrmf.entity_embeddings, self.lrmf.l2)
                loss += group_loss(
                    group, global_questions, local_questions, self.lrmf.entity_embeddings, self.lrmf.ratings)

            if loss < min_loss:
                min_loss = loss
                best_question = candidate

        self.question = best_question
        remaining_candidates = except_item(candidates, self.question)
        self.l2_questions = local_questions_vector(
            remaining_candidates, self.lrmf.entity_embeddings, self.lrmf.l2)

        self.transformation = transformation(
            self.users, group_representation(self.users, self.l1_questions, self.l2_questions, self.lrmf.ratings),
            self.lrmf.entity_embeddings, self.lrmf.ratings)

        self.children = {
            LIKE: Tree(self.l1_questions + [self.question], self.depth + 1, self.max_depth, self.lrmf),
            DISLIKE: Tree(self.l1_questions + [self.question], self.depth + 1, self.max_depth, self.lrmf)
        }

        likes, dislikes = split_users(self.users, self.question, self.lrmf.ratings)
        self.children[LIKE].grow(likes, remaining_candidates)
        self.children[DISLIKE].grow(dislikes, remaining_candidates)

    def interview_existing_user(self, user: int) -> np.ndarray:
        if self.is_leaf():
            user_vector, = group_representation([user], self.l1_questions, self.l2_questions, self.lrmf.ratings)
            return user_vector @ self.transformation

        answer = self.lrmf.ratings[user, self.question]
        if answer < 5:
            return self.children[1].interview_existing_user(user)
        elif answer <= 6:
            return self.children[5].interview_existing_user(user)

    def interview_new_user(self, actual_answers, user_answers) -> Union[int, np.ndarray]:
        if self.is_leaf():
            # Have we asked all our local questions?
            if len(user_answers) < self.lrmf.interview_length:
                for local_question in self.l2_questions:
                    # First try to exhaust the available answers
                    try:
                        answer = actual_answers[local_question]
                        u_a = user_answers.copy()
                        u_a.append(answer)
                        return self.interview_new_user(actual_answers, u_a)

                    # If we cannot get an answer from the arguments, return the question
                    except IndexError:
                        return local_question

            # If we have asked all of our questions, return the transformed user vector
            else:
                user_vector = [a for a in user_answers]
                user_vector.append(1)  # Add bias
                return user_vector @ self.transformation

#        if not actual_answers:
#            return self.question

        # find answer to global question
        answer = actual_answers[self.question]

        u_a = user_answers.copy()
        u_a.append(answer)
        if answer >= 4:
            return self.children[5].interview_new_user(actual_answers, u_a)
        if answer < 4:
            return self.children[1].interview_new_user(actual_answers, u_a)

class LRMFMain():
    def __init__(self, data, candidates):
        self.candidates = candidates
        self.data = data

        self.n_users = self.data.shape[0]
        self.n_entities = self.data.shape[1]

    def interview(self, l1, l2):
        self.model = LRMF(n_users=self.n_users, n_entities=self.n_entities,
                              l1=l1, l2=l2, kk=20, regularisation=0.01)  # See notes
        self._fit()


    def _fit(self, iterations=5):
#        for iteration in tqdm(range(0, iterations), desc=f'[Training LRMF]'):
#            self.model.fit(self.data, self.candidates)
#            tree = self.model.T
#           items = self.model.entity_embeddings

#            with open(f'models/tree_each_movie_loan_{iteration}.txt', 'wb') as f:
#                pickle.dump(tree, f)
#            with open(f'models/items_each_movie_loan_{iteration}.txt', 'wb') as f:
#               pickle.dump(items, f)

        with open(f'models/tree_each_movie_loan_{3}.txt', 'rb') as f:
            tree = pickle.load(f)
        with open(f'models/items_each_movie_loan_{3}.txt', 'rb') as f:
            items = pickle.load(f)

            # finds items in training that are not in test
        not_in_test = np.setdiff1d(train_iids, test_iids)

            # removes the items only in training based on their index in the train_iid list.
        for element in not_in_test:
            np.delete(items, train_iids.index(element), axis=0)

        test_users = range(test_data.shape[0])
        user_profiles = pd.DataFrame(data=0, index=test_users, columns=range(20))

        u_a_empty = []

        for user in test_users:
            user_profiles.iloc[user] = tree.interview_new_user(test_data[user], u_a_empty)

        reconstructed_ratings = user_profiles @ items.T

        m = eval.Metrics2(reconstructed_ratings.to_numpy(), test_data, 10, 'precision').calculate()

        print('test')


if __name__ == '__main__':
#    data = utils.load_data('eachmovie_triple').astype('int')
#    data.columns = ['iid', 'uid', 'count']
    #train_data, test_data = utils.train_test_split_user(data)

    #train_data.to_csv('data/each_movie_train_30_70.csv', sep=',', index=False)
    #test_data.to_csv('data/each_movie_test_30_70.csv', sep=',', index=False)

    train_data = pd.read_csv('data/each_movie_train_30_70.csv', sep=',')
    train_data = train_data.drop(train_data.columns[0], axis=1)
    test_data = pd.read_csv('data/each_movie_test_30_70.csv', sep=',')
    test_data = test_data.drop(test_data.columns[0], axis=1)

    train_data = pd.pivot_table(train_data, values='count', index='uid', columns='iid').fillna(0)
    test_data = pd.pivot_table(test_data, values='count', index='uid', columns='iid').fillna(0)

    train_iids = list(train_data.columns)
    test_iids = list(test_data.columns)

    train_uids = list(train_data.index)
    test_uids = list(test_data.index)

    train_data = train_data.to_numpy().astype('int')
    test_data = test_data.to_numpy().astype('int')

    with open(f'data/candidate_items_eachmovie.txt', 'rb') as f:
        candidate_set = pickle.load(f)

    lrmf = LRMFMain(train_data, candidate_set)
    lrmf.interview(3, 2)
    print('Heeej')
