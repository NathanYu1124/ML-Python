import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# 感知机: 训练数据集必须线形可分
class Perceptron:

    def __init__(self): 
        self.learning_rate = 0.1
        self.has_trained = False
        
    def sign(self, x, w, b):
        res = np.dot(x, w) + b
        return (1 if res >= 0 else -1)

    # 训练来获取w和b
    def train(self, X_train, y_train):
        
        self.has_trained = True
        
        # 参数w,b初始化
        self.w = np.ones(X_train.shape[1], dtype=np.float32)
        self.b = 0.0
        
        # 是否有误分类点
        has_wrong = True
        while has_wrong:
            wrong_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.learning_rate * np.dot(y, X)
                    self.b = self.b + self.learning_rate * y
                    wrong_count += 1
                    
            if wrong_count == 0:
                has_wrong = False
        print("训练完成")   
    
    # 用训练后的w和b来预测
    def predict(self, X_data):
        if self.has_trained:
            labels = []
            for x in X_data:
                labels.append(self.sign(x, self.w, self.b))
            return labels
        else:
            print("predict_Error: 请先训练数据以获取参数w和b")
    
    # 获取当前模型的正确率
    def get_correct_rate(self, validate_data, validate_labels):
        labels = self.predict(validate_data)
        same_count = np.sum(labels == validate_labels)
        return float(same_count) / len(validate_labels)
    
    
    # 获取感知机当前的w和b
    def get_current_para(self):
        if self.has_trained:
            return self.w, self.b
        else:
            print("Error: 请先训练数据以获取参数w和b") 


# 获取训练数据
def load_train_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
 
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])
    
    return X, y



# 数据分类可视化
def plot_data(X_data, w, b):
    x_points = np.linspace(4, 7, 10)
    y_hat = -(w[0] * x_points + b) / w[1]
    plt.plot(x_points, y_hat)

    plt.scatter(X_data[:50, 0], X_data[:50, 1], c='red', label='0')
    plt.scatter(X_data[50:100, 0], X_data[50:100, 1], c='green', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend();
    


X, y = load_train_data()
ppn = Perceptron()
ppn.train(X, y)
w, b = ppn.get_current_para()
plot_data(X, w, b)
