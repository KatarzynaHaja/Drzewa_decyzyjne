class Node:
    def __init__(self,atribiute=None, condition=None, parent=None,children=list()):
        self.atribiute = atribiute
        self.condition = condition
        self.parent = parent
        self.children=children