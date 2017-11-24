from Drzewa_decyzyjne.decision_tree import Decision_Tree
from Drzewa_decyzyjne.node import Node
import copy

class TreeBuilder:

    def __init__(self,file_name):
        self.decistion_tree=Decision_Tree(file_name)
        self.decistion_tree.import_dataset()
        self.decistion_tree.divide_set()

    def build_tree(self):
        node=Node()
        self.tree=node
        self.build_tree_recursive(self.decistion_tree.train_data,node)

    def build_tree_recursive(self,P,node):
        by_classes=self.decistion_tree.divide_by_classes(P)
        if len(by_classes.keys())==1:
            attr=next(iter(by_classes.keys()))
            node.atribute=attr
            return

        index=self.decistion_tree.find_the_best_atribiute(P)
        node.atribiute=index

        by_attr=self.decistion_tree.divide_by_attributes(P,index)
        size=len(by_attr.keys())

        for k in by_attr.keys():
            new_node=Node(condition=k,parent = node)
            node.children.append(new_node)
            self.build_tree_recursive(by_attr[k],new_node)

t = TreeBuilder("car.data")
t.build_tree()

print(t.tree.children[0].condition)