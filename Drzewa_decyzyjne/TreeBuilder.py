from Drzewa_decyzyjne.decision_tree import Decision_Tree
from Drzewa_decyzyjne.node import Node
import copy
from Drzewa_decyzyjne.Pruner import Pruner

class TreeBuilder:

    def __init__(self,file_name):
        self.decistion_tree=Decision_Tree(file_name)
        self.decistion_tree.import_dataset()
        self.decistion_tree.divide_set()

    def build_tree(self):
        node=Node()
        self.build_tree_recursive(self.decistion_tree.train_data,node)
        return node

    def build_tree_recursive(self,P,node):

        by_classes=self.decistion_tree.divide_by_classes(P)
        if len(by_classes.keys())==1:   #leaf
            node.attribute = next(iter(by_classes.keys()))
            node.local_P = P
            return

        index=self.decistion_tree.find_the_best_atribiute(P)
        node.attribute=index
        by_attr=self.decistion_tree.divide_by_attributes(P,index)
        for k in by_attr.keys():
            # print(count,":",index,": ",k)
            new_node=Node(condition=k,parent = node,local_P=by_attr[k])
            node.children.append(new_node)
            self.build_tree_recursive(by_attr[k],new_node)


    def traverse_tree(self,example,tree):
        copy_tree=copy.deepcopy(tree)
        while len(copy_tree.children) !=0:
            flag=0
            for j in copy_tree.children:
                # print(example[j.parent.attribute],":",j.condition)
                if example[j.parent.attribute] == j.condition:
                    flag=1
                    copy_tree = j
            if flag==0:
                return 0
        class_of_element = copy_tree.attribute
        if class_of_element == example[-1]:
            return 1
        else:
            return 0

    def traverse_all(self,tree,dataset):
        points =0
        for i in dataset:
            points +=self.traverse_tree(i,tree)
        print("accuracy",points/len(dataset))
        return points/len(dataset)

    def print_recursive(self,tree):
        if len(tree.children)==0:
            print(tree.children)
            return
        print(tree.attribute)
        for x in tree.children:
            self.print_recursive(x)



t = TreeBuilder("car.data")
tree = t.build_tree()
print("^initial_accuracy")
t.traverse_all(tree,t.decistion_tree.test_data)
p=Pruner(t,tree)
p.prune(tree)
tree=p.best_tree
print("^po pruningu dla testowego")
t.traverse_all(tree,t.decistion_tree.test_data)

print("alo")
#
# z = TreeBuilder("lenses.data")
# tree = z.build_tree()
# print("^initial_accuracy")
# p=Pruner(z,tree)
# p.prune(tree)
# tree=p.best_tree
# print("^po pruningu dla testowego")
# z.traverse_all(tree,z.decistion_tree.wal_data)
#
# print("alo")
