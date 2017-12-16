import numpy as np
import collections
from sklearn.datasets import load_iris
import pandas as pd
import pptree
data = load_iris()
y = data.target
X = data.data
da = np.insert(X,X.shape[1],y,axis=1)
da = np.random.permutation(da)
X = da[:2*int(da.shape[0]/3),:-1]
y = da[:2*int(da.shape[0]/3),-1]
X_predict = da[2*int(da.shape[0]/3)+1:,:-1]
y_real = da[2*int(da.shape[0]/3)+1:,-1]

class Node:

    def __init__(self, attribute=None, parent=None, condition=None,local_P = None, value=None, name = None):
        self.local_P = local_P
        self.attribute = attribute
        self.condition = condition
        self.parent = parent
        self.value = value
        self.name = name
        self.children = list()

    # def preorder(self):
    #     yield self.attribute
    #     for child in self.children:
    #         yield from child.preorder()

    def __str__(self):
            return self.attribute
class DecisionTree:

    def __init__(self, X, y,categorical =False, max_depth=20, min_size=2, train = 1, wal =1, names=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.X = X
        self.y = y
        self.names = names
        self.categorical = categorical
        self.data = np.insert(self.X,self.X.shape[1],self.y,axis=1)
        self.split_data(train,wal)
        self.build_tree()

    def split_data(self,train, wal):
        self.data = np.random.permutation(self.data)
        self.training = self.data[:int(self.data.shape[0]  *train / (train + wal) )]
        self.wal = self.data[self.training.shape[0]:]

    def entropy(self,P):
        information = 0
        values = self.divide_by_classes(P)
        for i in values.keys():
            information += len(values[i]) / len(P) * np.log2(len(values[i]) / len(P))
        return - information

    def conditional_entropy(self,P,t,index_treshold):
        entropy =0
        value = self.treshold(P,t,index_treshold)
        for i in value.keys():
            entropy+= len(value[i])/len(P) * self.entropy(value[i])
        return entropy

    def treshold(self,P,t,index):
        data = sorted(P,key= lambda x : x[t])
        return {'0':data[:index], '1':data[index+1:]}

    def find_the_best_threshold(self,P,t):
        gains = list()
        data = sorted(P, key = lambda x : x[t])
        for i in range(len(data)):
            gains.append(self.gain(data,t,i))
        return gains.index(max(gains))

    def gain(self,P,t,index_treshold):
        return self.entropy(P)-self.conditional_entropy(P,t,index_treshold)

    def find_the_best_atribiute(self, P):
        gain_list = list()
        treshold = list()
        for i in range(P.shape[1] - 1):
            index = self.find_the_best_threshold(P,i)
            treshold.append(index)
            gain_list.append(self.gain(P,i,index))
        data = np.array(sorted(P, key=lambda x: x[gain_list.index(max(gain_list))])).reshape(P.shape[0],P.shape[1])
        tr = treshold[gain_list.index(max(gain_list))]
        value = data[tr,gain_list.index(max(gain_list))]
        return [gain_list.index(max(gain_list)), value, tr ]

    def count_information_categorical(self,P):
        information = 0
        values = self.divide_by_classes(P)
        for i in values.keys():
            information += len(values[i]) / len(P) * np.log2(len(values[i]) / len(P))
        return - information

    def conditional_entropy_categorical(self,P,t):
        entropy =0
        value = self.divide_by_attributes_categorical(P,t)
        for i in value.keys():
            entropy+= len(value[i])/len(P) * self.count_information_categorical(value[i])
        return entropy

    def gain_categorical(self,P,t):
        return self.count_information_categorical(P)-self.conditional_entropy_categorical(P,t)

    def find_the_best_atribiute_categorical(self,P):
        gain_list = list()
        for i in range(P.shape[1]-1):
            gain_list.append(self.gain_categorical(P,i))

        return gain_list.index(max(gain_list))

    def divide_by_attributes_categorical(self, P, t):
        attr_set = set()
        for i in P:
            attr_set.add(i[t])

        by_attr=collections.defaultdict(list)
        for i in P:
            by_attr[i[t]].append(np.array(i))

        for k in by_attr.keys():
            by_attr[k]=np.array(by_attr[k])
        return by_attr


    def divide_by_classes(self, P):
        classes = collections.defaultdict(list)
        for j in P:
            classes[j[-1]].append(j)

        for k in classes.keys():
            classes[k] = np.array(classes[k])

        return classes

    def build_tree_recursive(self, P, node, max_depth, i, min_sample):
        by_classes = self.divide_by_classes(P)
        i += 1

        if len(by_classes.keys()) == 1 or i == max_depth or P.shape[0]  <= min_sample :
            node.attribute = next(iter(by_classes.keys()))
            node.name = next(iter(by_classes.keys()))
            node.local_P = P
            return

        if self.categorical == True:
            index = self.find_the_best_atribiute_categorical(P)
            node.attribute = index
            node.name = self.names[index]
            node.local_P = P
            by_attr = self.divide_by_attributes_categorical(P, index)
            for k in by_attr.keys():
                new_node = Node(condition=k, parent=node, local_P=by_attr[k])
                node.children.append(new_node)
                self.build_tree_recursive(by_attr[k], new_node, max_depth, i,min_sample)

        else:
            attr , value, treshold = self.find_the_best_atribiute(P)
            node.attribute = attr
            node.local_P = P
            node.value = value
            data = np.array(sorted(P, key=lambda x: x[attr])).reshape(P.shape[0],P.shape[1])
            if treshold == 0:
                treshold =1
            new_node_no = Node(condition = "no",parent=node, local_P=np.array(data[:treshold]))
            new_node_yes =  Node(condition = "yes",parent=node, local_P=np.array(data[treshold+1:]))
            node.children.append(new_node_no)
            node.children.append(new_node_yes)
            for j in node.children:
                self.build_tree_recursive(j.local_P, j, max_depth, i, min_sample)

    def build_tree(self):
        node = Node()
        self.build_tree_recursive(np.array(self.training), node, self.max_depth,-1, self.min_size)
        self.tree = node

    def traverse_tree(self,example):
        copy_tree = self.tree
        while len(copy_tree.children) !=0:
            if example[copy_tree.attribute] > copy_tree.value:
                copy_tree = copy_tree.children[1]
            else:
                copy_tree = copy_tree.children[0]

        return copy_tree.attribute

    def traverse_tree_categorical(self,example):
        copy_tree = self.tree
        while len(copy_tree.children) !=0:
            flag=0
            for j in copy_tree.children:
                if example[j.parent.attribute] == j.condition:
                    flag=1
                    copy_tree = j
            if flag==0:
                return "unacc"
        return copy_tree.attribute


    def predict(self, X):
        predicted_classes = list()
        if self.categorical == False:
            for i in X:
                predicted_classes.append(self.traverse_tree(i))
        else:
            for i in X:
                predicted_classes.append(self.traverse_tree_categorical(i))
        return np.array(predicted_classes)

    def count_accuracy(self,data):
        points =0
        predicted_class = self.predict(data[:,:-1])
        real_class = data[:,-1]
        for i in range(len(real_class)):
            if real_class[i] == predicted_class[i]:
                points+=1
        return points/len(predicted_class)

    def __str__(self):
        # if self.categorical == False:
        #     self.print_recursive(self.tree)
        # else:
        #     self.print_level(self.tree)
        pptree.print_tree(d.tree)




    def print_level(self,node):
        level = 0
        lastPrintedLevel = 0
        visit = []
        visit.append((node, level))
        parent = node
        while len(visit) != 0:
            item = visit.pop(0)
            if item[1] != lastPrintedLevel:  # New line for a new level
                lastPrintedLevel += 1
                print()
                print("Poziom : ",str(item[1]))
                print()
            if item[0].parent != None and parent.name != None:
                if parent != item[0].parent:
                    print("WEZEL***************************************")
            if item[0].parent != None:
                print("atrybut: ", item[0].name, "warunek: ",item[0].condition, "rodzic: ", item[0].parent.name)
            else:
                print("atrybut: ", item[0].name)

            for i in item[0].children:
                if i!= None:
                    visit.append((i, item[1] + 1))
            if item[0].parent != None:
                parent = item[0].parent

    def print_recursive(self, tree):
        if len(tree.children) == 0:
            print("atrybut: ",tree.attribute, "rodzic: ", tree.parent.attribute , "warunek:", tree.condition)
            return
        if tree.parent == None:
            print(tree.attribute, "> ", tree.value, "warunek:", tree.condition )
        else:
            print(tree.attribute, "> ", tree.value, "warunek:", tree.condition, "rodzic: ", tree.parent.attribute)
        for x in tree.children:
            self.print_recursive(x)


#
# blee = np.insert(X_predict,X_predict.shape[1],y_real,axis=1)
# d = DecisionTree(X,y,categorical=False)
# print("accuracy",d.count_accuracy(blee[:100]))
# d.__str__()

df = pd.read_csv("car.data")
data = np.array(df.values)
names = np.array(pd.read_csv("car.data", nrows=1).columns)
y = data[:, -1]
X = data[:,:-1]
da = np.insert(X,X.shape[1],y,axis=1)
da = np.random.permutation(da)
X = da[:2*int(da.shape[0]/3),:-1]
y = da[:2*int(da.shape[0]/3),-1]

X_predict = da[2*int(da.shape[0]/3)+1:,:-1]
y_real = da[2*int(da.shape[0]/3)+1:,-1]
blee = np.insert(X_predict,X_predict.shape[1],y_real,axis=1)
d = DecisionTree(X,y,True, names = names)
print("accuracy",d.count_accuracy(blee))
# print(d.tree.attribute,"> ", d.tree.value)
d.__str__()



