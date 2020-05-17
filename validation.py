import itertools
from random import shuffle
from typing import List


def k_fold_cross_validation(model, k: int, n: int, x: List[list],
                            y: list) -> float:
    pairs = list(zip(x, y))
    total_accuracy = 0.0

    for p in range(k):
        print(f'Validation iteration: {p + 1}/{k}')
        shuffle(pairs)
        subdivisions = list(_split(pairs, n))

        for i in range(n):
            print(f'    k-fold iteration: {i + 1}/{n}')
            test = subdivisions[i]
            train = [tmp for j, tmp in enumerate(subdivisions) if j != i]
            train = itertools.chain.from_iterable(train)  # flatten
            x_test, y_test = zip(*test)
            x_test, y_test = list(x_test), list(y_test)
            x_train, y_train = zip(*train)
            x_train, y_train = list(x_train), list(y_train)

            model.fit(x_train, y_train)
            total_accuracy += 1 - model.eval(x_test, y_test)

    return total_accuracy / (n * k)


def _split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
