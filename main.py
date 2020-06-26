# Autor: Michał Wójcik

import argparse
import numpy as np
from typing import Tuple
import random

from edt import EDT
from validation import k_fold_cross_validation


def read_data(dataset) -> Tuple[np.ndarray, np.ndarray]:
    path = f'datasets/{dataset}.csv'
    content = np.loadtxt(path, delimiter=',')
    X = content[:, :-1]
    y = content[:, -1]

    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['abalone', 'breast-tissue', 'ecoli', 'page-blocks', 'seeds',
                 'winequality-red', 'winequality-white'],
        default='seeds'
    )

    args = parser.parse_args()
    return args.dataset


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    dataset = parse_args()
    X, y = read_data(dataset)

    mi, lambda_, target_h, tournament_k, mutation_prob = 245, 75, 10, 5, 0.0325
    print(f'{mi}, {lambda_}, {target_h}, {tournament_k}, {mutation_prob:.4f}')

    tree = EDT(
        mi=mi,
        lambda_=lambda_,
        p_split=0.5,
        target_height=target_h,
        tournament_k=tournament_k,
        mutation_prob=mutation_prob,
        max_iter=500,
        stall_iter=100
    )

    accuracy = k_fold_cross_validation(tree, 1, 5, X, y)
    print(f'Total accuracy: {100 * accuracy:.3f}%')
