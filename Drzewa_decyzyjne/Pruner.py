from Drzewa_decyzyjne.decision_tree import Decision_Tree
from Drzewa_decyzyjne.node import Node
import copy

class Pruner:

    def __init__(self,tree_builder,tree):
        self.tree_builder=tree_builder
        self.best_tree=copy.deepcopy(tree)

        self.best_accuracy=self.tree_builder.traverse_all(tree,self.tree_builder.decistion_tree.test_data)

    def prune(self,node):

        if len(node.children) ==0:
            return

        children=node.children

        flag=1

        for child in children:
            if len(child.children)>0:
                flag=0
                break

        if flag==1:     #node has only leafs, lets delete them and transform node to leaf
            classes = dict()
            for child in children:
                if child.attribute in classes:
                    classes[child.attribute] += 1
                else:
                    classes[child.attribute] = 1

            node.children=list()
            node.attribute=self.select_best_class(classes)
            root=self.find_root(node)
            accuracy=self.tree_builder.traverse_all(root,self.tree_builder.decistion_tree.test_data)
            print(accuracy," : ",self.best_accuracy)
            if accuracy>self.best_accuracy-0.0001:
                print("^better")
                self.best_accuracy=accuracy
                self.best_tree=root
                self.prune(self.best_tree)

        else:
            for child in children:
                if len(child.children)>0:
                    node=child
                    self.prune(node)


    def select_best_class(self,classes):

        best_class=None;
        best_val=-1;
        for c in classes.keys():
            if classes[c]>best_val:
                best_class=c

        return best_class

    def find_root(self,node):
        while node.parent!=None:
            node=copy.deepcopy(node.parent)

        return node;