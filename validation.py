import numpy as np


def k_fold_cross_validation(model, n: int, k: int, X: np.ndarray,
                            y: np.ndarray) -> float:
    total_accuracy = 0.0

    for o in range(n):
        print(f'Validation iteration: {o + 1}/{n}')
        p = np.random.permutation(len(y))
        X_shuffled, y_shuffled = X[p], y[p]
        X_batches = np.array_split(X_shuffled, k)
        y_batches = np.array_split(y_shuffled, k)

        for i in range(k):
            print(f'    k-fold iteration: {i + 1}/{k} ', end='', flush=True)
            X_test, y_test = X_batches[i], y_batches[i]
            X_train = np.vstack([batch for j, batch in enumerate(X_batches) if j != i])
            y_train = np.hstack([batch for j, batch in enumerate(y_batches) if j != i])

            model.fit(X_train, y_train)
            accuracy = 1 - model.eval(X_test, y_test)
            print(f'accuracy {100 * accuracy:.3f}%')
            total_accuracy += accuracy

    return total_accuracy / (n * k)
