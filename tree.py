# Autor: Michał Wójcik

from node import Node


class Tree:
    def __init__(self, root: Node, value: float):
        self.root = root
        self.value = value

    def copy(self):
        return Tree(self.root.copy(), self.value)
