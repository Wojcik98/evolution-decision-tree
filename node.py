from random import choice, randrange, random, uniform
from typing import List, Tuple
import numpy as np


class Node:
    def __init__(self):
        self.parent: Node = None
        self.attribute = None
        self.threshold = None
        self.child_yes: Node = None
        self.child_no: Node = None
        self.label = None

    def copy(self):
        new = Node()
        new.attribute = self.attribute
        new.threshold = self.threshold
        new.label = self.label

        if new.label is None:
            new.child_yes = self.child_yes.copy()
            new.child_yes.parent = new
            new.child_no = self.child_no.copy()
            new.child_no.parent = new

        return new

    def height(self) -> int:
        if self.label is not None:
            return 1
        else:
            return 1 + max(self.child_yes.height(), self.child_no.height())

    def subnodes_count(self) -> int:
        if self.label:
            return 1
        else:
            return self.child_yes.subnodes_count() + self.child_no.subnodes_count()


def get_nth_subnode(root: Node, n: int) -> Node:
    tmp: List[Node] = [root]

    for _ in range(n):
        cur = tmp.pop()
        if cur.label is None:
            tmp.append(cur.child_no)
            tmp.append(cur.child_yes)

    return cur


def generate_subtree(p_split: float, attributes: int,
                     ranges: List[Tuple[float, float]], labels: np.ndarray,
                     depth: int = 1) -> Node:
    MAX_DEPTH = 100
    node = Node()

    if random() < p_split and depth < MAX_DEPTH:
        node.attribute = randrange(attributes)
        node.threshold = uniform(*ranges[node.attribute])
        node.child_yes = generate_subtree(p_split, attributes, ranges, labels,
                                          depth + 1)
        node.child_yes.parent = node
        node.child_no = generate_subtree(p_split, attributes, ranges, labels,
                                         depth + 1)
        node.child_no.parent = node
    else:
        node.label = choice(labels)

    return node
