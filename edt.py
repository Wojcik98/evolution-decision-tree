from random import choice, choices, randint, randrange, random, uniform
from typing import List, Tuple

from node import Node, generate_subtree, get_nth_subnode
from tree import Tree

EPSILON = 0.001


class EDT:
    """Evolutionary Decision Tree"""

    def __init__(
        self,
        mi: int = 500,
        lambda_: int = 400,
        p_split: float = 0.5,
        target_height: int = 9,
        tournament_k: int = 5,
        mutation_prob: float = 0.05,
        max_iter: int = 500,
        stall_iter: int = 50
    ):
        self.mi = mi
        self.lambda_ = lambda_ + (lambda_ % 2)  # make it even
        self.p_split = p_split
        self.target_height = target_height
        self.tournament_k = tournament_k
        self.mutation_prob = mutation_prob
        self.max_iter = max_iter
        self.stall_iter = stall_iter

        self.root: Node = None

    def eval(self, x: List[list], y: list) -> float:
        """Returns error of prediction of x given true values y."""
        return self.eval_from_node(self.root, x, y)

    def predict(self, x: list):
        """Predicts output for input x."""
        if self.root is None:
            raise Exception('Model not trained!')
        return self.predict_from_node(self.root, x)

    def fit(self, x: List[list], y: list) -> None:
        """Finds decision tree that tries to predict y given x."""
        attributes = len(x[0])
        ranges = []
        for i in range(attributes):
            vals = [tmp[i] for tmp in x]
            ranges.append((min(vals), max(vals)))
        labels = list(set(y))

        P = []
        for _ in range(self.mi):
            root = generate_subtree(self.p_split, attributes, ranges, labels)
            value = self.ga_fun(root, x, y)
            P.append(Tree(root, value))

        stop = False
        iter = 1
        stall_iter = 0
        current_best = P[0]
        best_val = current_best.value

        while not stop:
            # TODO diagnostics
            R = self.select(P)
            C = self.crossover(R, x, y)
            O = self.mutation(C, x, y, attributes, ranges, labels)
            P = self.replace(P, O)

            P.sort(key=lambda tree: tree.value)
            current_best = P[0]  # TODO argmin

            if abs(current_best.value - best_val) < EPSILON:
                stall_iter += 1
            else:
                stall_iter = 0
                best_val = current_best.value

            mean = sum(tmp.value for tmp in P) / len(P)
            print(f"Iteration {iter}: best ga value: {best_val:.5f}, best value: {self.eval_from_node(current_best.root, x, y):.5f}, mean: {mean:.3f}")
            if iter >= self.max_iter or stall_iter >= self.stall_iter:
                stop = True
            iter += 1

        self.root = current_best.root

    def select(self, P: List[Tree]) -> List[Tree]:
        R = []

        for _ in range(self.lambda_):
            rank = choices(P, k=self.tournament_k)
            rank.sort(key=lambda tree: tree.value)  # TODO argmin
            R.append(rank[0])

        return R

    def crossover(self, R: List[Tree], x, y) -> List[Tree]:
        pairs = [(R[i], R[i + 1]) for i in range(int(len(R) / 2))]
        C = []

        for a, b in pairs:
            first = get_nth_subnode(a.root, randint(1, a.root.subnodes_count()))
            second = get_nth_subnode(b.root, randint(1, b.root.subnodes_count()))

            if first.parent is not None and second.parent is not None:
                first.parent, second.parent = second.parent, first.parent

                if second is first.parent.child_yes:
                    first.parent.child_yes = first
                else:
                    first.parent.child_no = first

                if first is second.parent.child_yes:
                    second.parent.child_yes = second
                else:
                    second.parent.child_no = second

            C.append(Tree(a.root, self.ga_fun(a.root, x, y)))
            C.append(Tree(b.root, self.ga_fun(b.root, x, y)))

        return C

    def mutation(self, C: List[Tree], x, y, attributes: int,
                     ranges: List[Tuple[float]], labels: list) -> List[Tree]:
        O = []

        for tree in C:
            tmp: List[Node] = [tree.root]
            while len(tmp) > 0:
                cur = tmp.pop()

                if random() < self.mutation_prob:
                    if cur.label is None:
                        cur.attribute = randrange(attributes)
                        cur.threshold = uniform(*ranges[cur.attribute])
                    else:
                        cur.label = choice(labels)

                if cur.label is None:
                    tmp.append(cur.child_no)
                    tmp.append(cur.child_yes)

            O.append(Tree(tree.root, self.ga_fun(tree.root, x, y)))

        return O

    def replace(self, P: List[Tree], O: List[Tree]) -> List[Tree]:
        union = P + O
        union.sort(key=lambda tree: tree.value)
        return union[:self.mi]

    def predict_from_node(self, root: Node, x: list):
        node = root
        while node.label is None:
            if x[node.attribute] > node.threshold:
                node = node.child_yes
            else:
                node = node.child_no

        return node.label

    def eval_from_node(self, root: Node, x: List[list], y: list) -> float:
        assert len(x) == len(y)
        preds = [self.predict_from_node(root, sample) for sample in x]
        errors = [pred != goal for pred, goal in zip(preds, y)]

        return sum(errors) / len(errors)

    def ga_fun(self, root: Node, x: List[list], y: list) -> float:
        error_factor = 0.99 * self.eval_from_node(root, x, y)
        height_factor = 0.01 * root.height() / self.target_height
        return error_factor + height_factor
