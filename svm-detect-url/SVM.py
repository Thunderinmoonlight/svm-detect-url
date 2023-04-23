# coding: utf-8
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
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
import csv
import pickle
import urllib

k = 80
# weather use kmean
use_k = True
use_SVD = False


class WAF(object):

    def Train(self):
        good_query_list = self.get_query_list('good_queries.txt')
        bad_query_list = self.get_query_list('bad_queries.txt')

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

        print('向量化后维度：' + str(X.shape))

        if use_SVD:
            lsa = TruncatedSVD(80)
            X = lsa.fit_transform(X)
            print('向量化后维度：' + str(X.shape))

        if use_k:
            print(self.kmeans(X),"\n--------------------------------------------")
            X = self.transform(self.kmeans(X))
            print(X)
            print('降维完成')
            print('降维后维度：' + str(X.shape))

        # 使用 train_test_split 分割 X y 列表
        # X_train矩阵的数目对应 y_train列表的数目(一一对应)  -->> 用来训练模型
        # X_test矩阵的数目对应 	 (一一对应) -->> 用来测试模型的准确性
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 定义svm方法模型
        self.clf = svm.SVC()

        # 使用svm方法训练模型实例 clf
        self.clf.fit(X_train, y_train)

        # 使用测试值 对 模型的准确度进行计算
        print('训练模型的准确度:{}'.format(self.clf.score(X_test, y_test)))

        # 保存训练结果
        with open('SVM.pickle', 'wb') as output:
            pickle.dump(self, output)

        #验证集检验模型精度
        with open('good_queries.txt', 'r') as f_1:
            preicdtlist_1 = [i.strip('\n') for i in f_1.readlines()[:]]
        with open('bad_queries.txt', 'r') as f_2:
            preicdtlist_2 = [i.strip('\n') for i in f_2.readlines()[:]]
        result_1 = self.check_model(preicdtlist_1, 0)
        result_2 = self.check_model(preicdtlist_2, 1)
        result = result_1 + result_2
        print(result)
        print('模型的准确度:{}'.format(result / (len(preicdtlist_1) + len(preicdtlist_2))))

    def kmeans(self, weight):

        print('kmeans之前矩阵大小： ' + str(weight.shape))
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('1.label', 'r') as input:

                print('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]

        except FileNotFoundError:

            print('Start Kmeans ')

            clf = KMeans(n_clusters=k, precompute_distances=False)

            s = clf.fit(weight)

            # 保存聚类的结果
            self.label = clf.labels_

            with open('1.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        print('kmeans 完成,聚成 ' + str(k) + '类')
        print('kmeans之后矩阵大小： ' + str(weight.shape))
        return weight

    def SVD(self, weight):

        print('kmeans之前矩阵大小： ' + str(weight.shape))
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('2.label', 'r') as input:

                print('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]

        except FileNotFoundError:

            print('Start SVD ')

            clf = KMeans(n_clusters=k, precompute_distances=False)

            s = clf.fit(weight)

            # 保存聚类的结果
            self.label = clf.labels_

            with open('2.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        print('kmeans 完成,聚成 ' + str(k) + '类')
        print('kmeans之后矩阵大小： ' + str(weight.shape))
        return weight

    #     转换成聚类后结果 输入转置后的矩阵 返回转置好的矩阵
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

    # 对 新的请求列表进行预测
    def predict(self, new_queries):
        try:
            with open('SVM.pickle', 'rb') as input:
                self = pickle.load(input)
            print('loading SVM model success')
        except FileNotFoundError:
            print('start to train the SVM model')
            self.Train()
        print('start predict:')

        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        if use_k:
            print('将输入转换')
            X_predict = self.transform(X_predict.tolil().transpose())
            # print(X_predict,"---\n")

        if use_SVD:
            print('将输入转换')
            lsa = TruncatedSVD(80)
            X_predict = lsa.fit_transform(X_predict.tolil().transpose())



        print('转换完成,开始预测')

        res = self.clf.predict(X_predict)
        res_list = []
        #  print("预测的结果列表:\n")
        for q, r in zip(new_queries, res):
            tmp = '正常请求' if r == 0 else '恶意请求'
            #   print('{}  {}'.format(q, tmp))
            q_entity = html.escape(q)
            res_list.append({q_entity: tmp})
        # print("预测的结果列表:{}".format(str(res_list)))
        return res_list

    # 验证集验证函数
    def check_model(self, new_queries, flag):
        with open('SVM.pickle', 'rb') as input:
            self = pickle.load(input)

        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        if use_k:
            X_predict = self.transform(X_predict.tolil().transpose())

        if use_SVD:
            lsa = TruncatedSVD(k)
            X_predict = lsa.fit_transform(X_predict)

        res = self.clf.predict(X_predict)
        res_list = []
        count = 0
        #  print("预测的结果列表:\n")
        for q, r in zip(new_queries, res):
            if r == 0 and flag == 0:
                count = count + 1
            elif r != 0 and flag == 1:
                count = count + 1

        return count

    # 得到文本中的请求列表
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

    # tokenizer function, this will make 3 grams of each query
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - 3):
            ngrams.append(tempQuery[i:i + 3])

        # print(ngrams)
        return ngrams


def save_file():
    fname = input('文件名为')
    out_file = open(fname, 'w')


s = WAF()
s.predict(['http://google.com', 'http://paypal-manager-loesung.net/konflikt/66211165125/'])
print(s.predict(['http://google.com', 'http://paypal-manager-loesung.net/konflikt/66211165125/']))
# root = tk.Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename()
# print(file_path)
# # testfile= 'test.txt'
# with open(file_path, 'r') as f:
#     #  print('预测数据集： '+testfile)
#     preicdtlist = [i.strip('\n') for i in f.readlines()[:]]
#     result = s.predict(preicdtlist)
#     print(result)
#
#     csv_file = open('result.csv', 'w', newline='')
#     writer = csv.writer(csv_file)
#     for I in result:
#         for key, value in I.items():
#             print(key, value)
#             writer.writerow([key, value])
#
#     pass
# with open('SVM.pickle','rb') as input:
#    w = pickle.load(input)
