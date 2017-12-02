class Node:

    def __init__(self, attribute=None, condition=None, parent=None, local_P = None, root = None):
        self.local_P = local_P
        self.attribute = attribute
        self.condition = condition
        self.parent = parent
        self.root = root
        self.children = list()
