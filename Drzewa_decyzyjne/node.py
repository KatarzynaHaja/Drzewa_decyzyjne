class Node:

    def __init__(self, attribute=None, condition=None, parent=None, local_P = None):
        self.local_P = local_P
        self.attribute = attribute
        self.condition = condition
        self.parent = parent
        self.children = list()
