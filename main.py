import numpy as np
from typing import Tuple

from edt import EDT
from validation import k_fold_cross_validation


def read_seeds() -> Tuple[np.ndarray, np.ndarray]:
    path = 'datasets/seeds/seeds_dataset.txt'
    x = []
    y = []

    with open(path, 'r') as file:
        for row in file.readlines():
            vals = row.split()
            x.append([float(tmp) for tmp in vals[:7]])
            y.append(vals[7])

    return np.array(x), np.array(y)


if __name__ == '__main__':
    x, y = read_seeds()
    tree = EDT(
        mi=100,
        lambda_=30,
        p_split=0.5,
        target_height=9,
        tournament_k=5,
        mutation_prob=0.005,
        max_iter=500,
        stall_iter=20
    )

    accuracy = k_fold_cross_validation(tree, 3, 5, x, y)
    print(f'Total accuracy: {100 * accuracy:.3f}%')
