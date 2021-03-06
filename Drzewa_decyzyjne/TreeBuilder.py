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
        self.build_tree_recursive(self.decistion_tree.train_data,node,5,0)
        return node

    def build_tree_recursive(self,P,node,max_depth,i):

        by_classes=self.decistion_tree.divide_by_classes(P)
        i+=1
        print(i)
        if len(by_classes.keys())==1 or i == max_depth:   #leaf
            node.attribute = next(iter(by_classes.keys()))
            node.local_P = P
            return

        index=self.decistion_tree.find_the_best_atribiute(P)
        node.attribute=index
        node.local_P = P
        by_attr=self.decistion_tree.divide_by_attributes(P,index) # tutaj zrobić tak żeby dzieliło na podzial
        for k in by_attr.keys():
            # print(count,":",index,": ",k)
            new_node=Node(condition=k,parent = node,local_P=by_attr[k])
            node.children.append(new_node)
            self.build_tree_recursive(by_attr[k],new_node, max_depth,i)


    def traverse_tree(self,example,tree):
        #copy_tree = copy.deepcopy(tree)
        copy_tree = tree
        while len(copy_tree.children) !=0:
            flag=0
            for j in copy_tree.children:
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
        copy_tree = copy.deepcopy(tree)
        for i in dataset:
            copy_tree = copy.copy(tree)
            points +=self.traverse_tree(i,copy_tree)
        return points/len(dataset)

    def print_recursive(self,tree):
        if len(tree.children)==0:
            print(tree.children)
            return
        print(tree.attribute)
        for x in tree.children:
            self.print_recursive(x)

    def clone (self,node):
        return node


print("testy dla zbioru cars")
t = TreeBuilder("car.data")
tree = t.build_tree()
print("^accuracy dla walidacyjnego")
print(t.traverse_all(tree,t.decistion_tree.wal_data))
print("^accuracy dla testowego na początku")
print(t.traverse_all(tree,t.decistion_tree.test_data))
p_tree = copy.deepcopy(tree)
p=Pruner(t,p_tree)
p.prune(p_tree)
tree=p.best_tree
print("^po pruningu dla testowego")
print(t.traverse_all(tree,t.decistion_tree.test_data))
#
# print("alo")
#
# z = TreeBuilder("lenses.data")
# tree = z.build_tree()
# print("^initial_accuracy")
# print(z.traverse_all(tree,z.decistion_tree.test_data))
# p=Pruner(z,tree)

print("testy dla zbioru balance")
z = TreeBuilder("balance1.data")
tree = z.build_tree()
init_tree = copy.deepcopy(tree)
train_tree = copy.deepcopy(tree)
print("^accuarcy dla treningowego")
print(z.traverse_all(train_tree,z.decistion_tree.train_data))
print("^accuracy na początku dla testowego")
print(z.traverse_all(init_tree,z.decistion_tree.test_data))
p_tree = copy.deepcopy(tree)
p=Pruner(z,p_tree)
p.prune(p_tree)
tree=p.best_tree
print("^po pruningu dla testowego")
print(z.traverse_all(tree,z.decistion_tree.test_data))

print("alo")