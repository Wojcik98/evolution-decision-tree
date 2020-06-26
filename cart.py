import numpy as np
import random
from sklearn import tree

from main import parse_args, read_data
from validation import k_fold_cross_validation


class CartTree(tree.DecisionTreeClassifier):
    def eval(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        errors = [pred == goal for pred, goal in zip(preds, y)]
        return sum(errors) / len(errors)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    dataset = parse_args()
    X, y = read_data(dataset)

    model = CartTree()
    accuracy = k_fold_cross_validation(model, 1, 5, X, y)
    print(f'Total accuracy: {100 * accuracy:.3f}%')
