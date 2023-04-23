# coding: utf-8
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import urllib
from sklearn import svm
import time
import pickle
import html
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import pickle
import urllib

k = 80
# weather use kmean
# use_k = False
use_k = True

class WAF(object):
    def Train(self):

        good_query_list = self.get_query_list('check_good_queries.txt')
        count = len(good_query_list)
        print(count)
        bad_query_list = self.get_query_list('check_bad_queries.txt')
        count = len(bad_query_list)
        print(count)

        # 打标记
        good_y = [0 for i in range(0, len(good_query_list))]
        bad_y = [1 for i in range(0, len(bad_query_list))]


        queries = bad_query_list + good_query_list
        y = bad_y + good_y

        # converting data to vectors  定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 把不规律的文本字符串列表转换成规律的 ( [i,j],tdidf值) 的矩阵X
        # 用于下一步训练分类器 lgs

        X = self.vectorizer.fit_transform(queries)
        lsa = TruncatedSVD(k)
        newX =lsa.fit_transform(X)
        print(newX)

        print('向量化后维度：' + str(newX.shape))

        X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.2, random_state=42)

        # 定义svm方法模型
        self.clf = svm.SVC()

        # 使用svm方法训练模型实例 clf
        self.clf.fit(X_train, y_train)

        # 使用测试值 对 模型的准确度进行计算
        print('模型的准确度:{}'.format(self.clf.score(X_test, y_test)))

        # 保存训练结果
        with open('SVM.pickle', 'wb') as output:
            pickle.dump(self, output)



        with open('good_queries.txt', 'r') as f_1:
            preicdtlist_1 = [i.strip('\n') for i in f_1.readlines()[:]]
        print(len(preicdtlist_1))
        with open('bad_queries.txt', 'r') as f_2:
            preicdtlist_2 = [i.strip('\n') for i in f_2.readlines()[:]]
        result_1 = self.check_model(preicdtlist_1, 0)
        result_2 = self.check_model(preicdtlist_2, 1)
        result = result_1+result_2
        print(result)
        print('模型的准确度:{}'.format(result/(len(preicdtlist_1)+len(preicdtlist_2))))

    def get_query_list(self, filename):
        directory = str(os.getcwd())
        # directory = str(os.getcwd())+'/module/waf'
        filepath = directory + "/" + filename
        data = open(filepath, 'r', encoding='UTF-8').readlines()
        query_list = []
        for d in data:
            d = str(urllib.parse.unquote(d))  # converting url encoded data to simple string
            # print(d)
            query_list.append(d)
        return list(set(query_list))

    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - 3):
            ngrams.append(tempQuery[i:i + 3])
        return ngrams

    def check_model(self, new_queries, flag):
        with open('SVM.pickle', 'rb') as input:
            self = pickle.load(input)

        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        if use_k:
            X_predict = self.transform(X_predict.tolil().transpose())
        res = self.clf.predict(X_predict)
        res_list = []
        count = 0
        #  print("预测的结果列表:\n")
        for q, r in zip(new_queries, res):
            if r == 0 and flag == 0:
                count = count + 1
            elif r != 0 and flag == 1:
                count = count + 1
        print(count)
        return count

    def transform(self, weight):

        from scipy.sparse import coo_matrix

        a = set()
        # 用coo存 可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号 label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        # print(row)
        # print(col)
        # print(data)
        newWeight = coo_matrix((data, (row, col)), shape=(k, weight.shape[1]))
        return newWeight.transpose()

if __name__ == '__main__':
    s=WAF()
    s.Train()
    # 验证集检验模型精度











