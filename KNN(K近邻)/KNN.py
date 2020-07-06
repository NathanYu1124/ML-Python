#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:10:53 2020

@author: nathanyu
"""

import math
import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



# KNN分类算法 
class KNNClassifier:
    
    # 初始化, 默认k值为5
    def __init__(self, k=5):
        self.k = k
        
    # 训练函数: KNN算法没有训练过程        
    def train(self, X_train, y_train):
        self.xtrain = X_train
        self.y = y_train
    
    # 预测函数
    def predict(self, x):
        idx = np.argsort([np.linalg.norm(x - x_sample) for x_sample in self.xtrain])[:self.k]
        knn_y = self.y[idx]
        vote = defaultdict(int)
        for y_pred in knn_y:
            vote[y_pred] += 1
        vote_list = list(vote.items())
        vote_list.sort(key=lambda x: x[1], reverse=True)
        return vote_list[0][0]
        
    
        
# 测试
def main():
    X, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = KNNClassifier()
    model.train(xtrain, ytrain)

    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            print("该样本真实标签为：{}，但是KNN分类模型预测标签为：{}".format(ytest[i], y_pred))
    print("KNN分类模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))
    print(n_right, n_test)

if __name__ == "__main__":
    main()