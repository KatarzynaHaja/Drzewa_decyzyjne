import pandas as pd
import numpy as np
import random
import collections
import itertools


class Decision_Tree:
    def __init__(self, file):
        self.file = file

    def import_dataset(self):
        df = pd.read_csv(self.file, header=0)
        self.data = np.array(df.values)

        y = self.data[:, -1]
        self.classes = set(y)

    def divide_set(self, tr=5, wal=1, test=1):
        self.data = np.random.permutation(self.data)

        self.train_data = self.data[:int(self.data.shape[0]  *tr / (tr + wal + test) )]
        self.wal_data = self.data[
                        self.train_data.shape[0] + 1:int(self.train_data.shape[0] + int(self.data.shape[0] / (tr + wal + test)))]
        self.test_data = self.data[self.train_data.shape[0] + self.wal_data.shape[0]+1:]

    def count_information(self,P):
        information = 0
        values = self.divide_by_classes(P)
        for i in values.keys():
            information += len(values[i]) / len(P) * np.log2(len(values[i]) / len(P))
        return - information

    def conditional_entropy(self,P,t):
        entropy =0
        value = self.divide_by_attributes(P,t)
        for i in value.keys():
            entropy+= len(value[i])/len(P) * self.count_information(value[i])
        return entropy

    def gain(self,P,t):
        return self.count_information(P)-self.conditional_entropy(P,t)

    def find_the_best_atribiute(self,P):
        gain_list = list()
        for i in range(P.shape[1]-1):
            gain_list.append(self.gain(P,i))

        return gain_list.index(max(gain_list))


    def divide_by_classes(self, P):
        classes = collections.defaultdict(list)
        for j in P:
            classes[j[-1]].append(j)

        for k in classes.keys():
            classes[k]=np.array(classes[k])

        return classes

    def divide_by_attributes(self, P, t):
        attr_set = set()
        for i in P:
            attr_set.add(i[t])

        by_attr=collections.defaultdict(list)
        for i in P:
            by_attr[i[t]].append(np.array(i))

        for k in by_attr.keys():
            by_attr[k]=np.array(by_attr[k])
        return by_attr


    def get_by_attr_val(self,P,t,r):
        elements=self.divide_by_attributes(P,t)
        return elements[r]

    def get_by_attr_val_and_class(self,P,t,r):
        elements=self.get_by_attr_val(P,t,r)
        return self.divide_by_classes(elements)



    def divide_by_attributes_all(self,P):
        attr_list=list()
        size = len(P[0])
        for i in range(0,size-1):
            attr_list.append(set())

        for i in range(0, size-1):
            for j in P:
                attr_list[i].add(j[i])
        return attr_list


#
# d = Decision_Tree("blee.data")
# d.import_dataset()
# d.divide_set(1,0,0)
# print(len(d.train_data))
#print(d.divide_by_classes(d.data))
# #print(d.divide_by_attributes(d.train_data, 1)['high'][:,2])
# #print(d.divide_by_attributes_all(d.train_data))
# #print(d.get_by_attr_val(d.train_data,1,'high'))
# #print(d.get_by_attr_val_and_class(d.train_data,1,'high'))
#print(d.count_information(d.train_data))
#print(d.conditional_entropy(d.data),1)
# print(d.gain(d.train_data,1))
#print(d.find_the_best_atribiute(d.train_data))