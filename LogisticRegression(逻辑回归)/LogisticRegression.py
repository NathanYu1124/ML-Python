#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:47:31 2020

@author: nathanyu
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 逻辑回归模型
class LogisticRegression():
    
    
    # 初始化
    def __init__(self, epoch_num, learning_rate, loss_tolerance):
        # 最大迭代次数-用于随机梯度下降法, 
        self.epoch_num = epoch_num   
        # 学习率-用于随机梯度下降法, 范围为 (0, 1]
        self.learning_rate = learning_rate
        # 误差范围-用于随机梯度下降法, 当两次迭代间的误差小于此范围时,退出迭代
        self.loss_tolerance = loss_tolerance
        
    # 模型训练
    def train(self, X, y):
        # 样本个数(行数), 样本的特征个数(列数)
        n_sample, n_feature = X.shape
        
        # 用于生成随机数
        rand_val = 1 / np.sqrt(n_feature)
        rng = np.random.default_rng()
        
        # 模型参数初始化
        self.w = rng.uniform(-rand_val, rand_val, size=n_feature)
        self.b = 0
        
        n_epoch = 0     # 当前epoch的迭代次数
        pre_loss = 0    # 上一轮的误差
        
        while True:
            cur_loss = 0    # 当前轮的误差
            
            # 遍历所有样本
            for i in range(n_sample):
                # LR模型给出的预测值
                y_pred = self.sigmoid(np.dot(self.w, X[i]) + self.b)
                # 预测值与实际值间的偏差
                y_diff = y[i] - y_pred
                # 随机梯度下降来更新参数w和b
                self.w += self.learning_rate * y_diff * X[i]
                self.b += self.learning_rate * y_diff
                # 累加当前轮的误差
                cur_loss += abs(y_diff)
            # 迭代次数+1
            n_epoch += 1
            # 和上一轮间的误差
            loss_diff = abs(cur_loss - pre_loss)
            # 更新上一轮的误差
            pre_loss = cur_loss
            
            # 迭代次数超过最大值或误差小于最小值, 退出循环
            if n_epoch >= self.epoch_num or loss_diff < self.loss_tolerance:
                break
    # 预测函数
    def predict(self, x):
        logit = np.dot(self.w, x) + self.b
        return 1 if logit > 0 else 0
    
    # sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
# 测试
def main():    
    
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X[:100], y[:100], train_size=0.8, shuffle=True)
        
    model = LogisticRegression(500, 0.01, 0.0001)
    model.train(x_train, y_train)
        
    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(x_test[i])
        if y_pred == y_test[i]:
            n_right += 1
        else:
            print("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(y_test[i], y_pred))
    print("LogisticRegression模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))
    print(n_right, n_test)

if __name__ == "__main__":
    main()
                
                
