import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as select
from sklearn import svm
import pandas as pd
import sklearn.linear_model as reg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class parser(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

    def split_data(self):
        data = np.asarray(self.get_input())
        x_array = data[:, [0, 1]].astype(np.float)
        y_array = data[:, 2].astype(np.int)
        X_train, X_test, y_train, y_test = select.train_test_split(x_array,
                                                                   y_array, test_size=0.4, train_size=0.6,
                                                                   stratify=y_array)
        return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.split_data()
        parameters = {
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                          50], "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        svc = RandomForestClassifier()
        clf = select.GridSearchCV(svc, parameters, cv=5)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        print(clf.best_score_)

    def test(self):
        X_train, X_test, y_train, y_test = self.split_data()
        clf = RandomForestClassifier(max_depth=9, min_samples_split=7)
        clf.fit(X_train, y_train)
        results = clf.predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == results[i]:
                correct += 1
        print(correct / len(y_test))

    def plot(self):
        data = np.asarray(self.get_input())
        x = data[:, 0].astype(np.float)
        y = data[:, 1].astype(np.float)
        labels = data[:, 2]
        labels = labels.astype(np.int)
        color = ["red" if l == 0 else "blue" for l in labels]
        plt.scatter(x, y, c=color)
        plt.show()

    def get_input(self):
        example_list = []
        with open(self.infile, "r") as r:
            reader = csv.reader(r)
            header = next(reader)
            for row in reader:
                example_list.append(row)
        return example_list


obj = parser("input3.csv", "output3.csv")
# obj.train()
obj.test()