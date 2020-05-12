from random import choice, randrange, random, uniform
from typing import List, Tuple


class Node:
    def __init__(self):
        self.parent: Node = None
        self.attribute = None
        self.threshold = None
        self.child_yes: Node = None
        self.child_no: Node = None
        self.label = None

    def height(self) -> int:
        if self.label:
            return 1
        else:
            return max(self.child_yes.height(), self.child_no.height())

    def subnodes_count(self) -> int:
        if self.label:
            return 1
        else:
            return self.child_yes.subnodes_count() + self.child_no.subnodes_count()


def get_nth_subnode(root: Node, n: int) -> Node:
    tmp: List[Node] = [root]    # TODO rename

    for _ in range(n):
        cur = tmp.pop()
        if cur.label is None:
            tmp.append(cur.child_no)
            tmp.append(cur.child_yes)

    return cur


def generate_subtree(p_split: float, attributes: int,
                     ranges: List[Tuple[float]], labels: list) -> Node:
    # TODO max depth
    node = Node()
    if random() < p_split:
        node.attribute = randrange(attributes)
        node.threshold = uniform(*ranges[node.attribute])
        node.child_yes = generate_subtree(p_split, attributes, ranges, labels)
        node.child_yes.parent = node
        node.child_no = generate_subtree(p_split, attributes, ranges, labels)
        node.child_no.parent = node
    else:
        node.label = choice(labels)
    return node
