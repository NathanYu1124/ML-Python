#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:56:29 2020

@author: nathanyu
"""

import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 朴素贝叶斯
class NaiveBayes():
    # 初始化
    def __init__(self):
        # 存储先验概率 P(Y = ck)
        self.prior_prob = defaultdict(float)
        # 存储似然概率 P(X|Y=ck)
        self.likelihood = defaultdict(defaultdict)
        # 存储每个类别的样本在训练集中的出现次数
        self.ck_counter = defaultdict(float)
        # 存储每一个特征可能取值的个数
        self.Sj = defaultdict(float)
        
        
    # 模型训练, 参数估计使用贝叶斯估计
    def train(self, X_train, y_train):
        # 样本个数(行数), 样本中的特征个数(列数)
        n_sample, n_feature = X_train.shape
        
        # 样本的类别(升序数组), 对应类别的样本个数(数组)
        ck, num_ck = np.unique(y_train, return_counts=True)
        # 样本类别dict: key-类别, value-对应类别样本个数
        self.ck_counter = dict(zip(ck, num_ck))
        
        # 贝叶斯估计: 计算先验概率, 做拉普拉斯平滑处理
        for label, num_label in self.ck_counter.items():
            self.prior_prob[label] = (num_label + 1) / (n_sample + ck.shape[0])
        
        # 记录每个类别对应的样本在训练集中的索引数组
        ck_idx = []
        for label in ck:
            label_idx = np.squeeze(np.argwhere(y_train == label))
            ck_idx.append(label_idx)
        
        # 遍历每个类别
        for label, idx in zip(ck, ck_idx):
            xdata = X_train[idx]
            
            # 记录类别为label时, 每个特征对应的概率 
            label_likelihood = defaultdict(defaultdict)
            
            # 遍历每个特征
            for i in range(n_feature):
                # 记录该特征每个取值对应的概率
                feature_val_prob = defaultdict(float)
                # 第i个特征的所有取值(升序数组), 对应取值的样本个数(数组)
                feature_val, feature_cnt = np.unique(xdata[:, i], return_counts=True)
                # 记录第i个特征的可能取值个数  
                self.Sj[i] = feature_val.shape[0]
                feature_counter = dict(zip(feature_val, feature_cnt))
                for fea_val, cnt in feature_counter.items():
                    # 贝叶斯估计: 计算该列特征每个取值的概率, 做拉普拉斯平滑
                    feature_val_prob[fea_val] = (cnt + 1) / (self.ck_counter[label] + self.Sj[i])
                label_likelihood[i] = feature_val_prob
            self.likelihood[label] = label_likelihood
        
    # 预测函数: 计算后验概率时对概率取对数, 概率连乘可能导致浮点数下溢, 取对数将连乘转化为求和
    def predict(self, X_validate):
        # 保存分类到每个类别的后验概率
        post_prob = defaultdict(float)
        # 遍历每个类别计算后验概率
        for label, label_likelihood in self.likelihood.items():
            prob = np.log(self.prior_prob[label])
            # 遍历样本的每一维特征
            for i, fea_val in enumerate(X_validate):
                feature_val_prob = label_likelihood[i]
                
                # 如果该特征值出现在训练集中则直接获取概率
                if fea_val in feature_val_prob:
                    prob += np.log(feature_val_prob[fea_val])
                else:
                # 如果该特征没有出现在训练集中则采用拉普拉斯平滑计算概率
                    laplace_prob = 1 / (self.ck_counter[label] + self.Sj[i])
                    prob += np.log(laplace_prob)
            post_prob[label] = prob
        prob_list = list(post_prob.items())
        prob_list.sort(key=lambda v: v[1], reverse=True)
        # 返回后验概率最大的类别作为预测类别
        return prob_list[0][0]
        
# 测试
def main():    
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
        
    model = NaiveBayes()
    model.train(x_train, y_train)
        
    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(x_test[i])
        if y_pred == y_test[i]:
            n_right += 1
        else:
            print("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(y_test[i], y_pred))
    print("Scratch模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))
    print(n_right, n_test)

if __name__ == "__main__":
    main()
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                