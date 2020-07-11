
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 感知机
class Perceptron():
    
    # 初始化
    def __init__(self, max_iter=500, learning_rate=0.1, loss_tolerance=0.001):
        # 最大迭代次数-用于随机梯度下降
        self.max_iter = max_iter
        # 学习率-用于随机梯度下降, 范围为(0, 1]
        self.learning_rate = learning_rate
        # 最小误差-用于随机梯度下降
        self.loss_tolerance = loss_tolerance
        
    # 模型训练
    def train(self, X, y):
        # 样本个数(行数), 样本的特征个数(列数)
        n_sample, n_feature = X.shape
        
        # 用于随机数生成
        rnd_val = 1 / np.sqrt(n_feature)
        rng = np.random.default_rng()
    
        # 初始化模型参数w和b
        self.w = rng.uniform(-rnd_val, rnd_val, size=n_feature)
        self.b = 0
        
        cur_iter = 0    # 当前循环次数
        pre_loss = 0    # 上一轮的损失
        
        while True:
            cur_loss = 0    # 当前轮的损失
            wrong_num = 0   # 误分类的样本数
            # 遍历所有样本
            for i in range(n_sample):
                y_pred = np.dot(self.w, X[i]) + self.b
                cur_loss += -y[i] * y_pred
                
                # 随机梯度下降法来更新参数
                if y[i] * y_pred <= 0:
                    self.w += self.learning_rate * X[i] * y[i]
                    self.b += self.learning_rate * y[i]
                    wrong_num += 1
                    
            cur_iter += 1
            loss_diff = cur_loss - pre_loss
            pre_loss = cur_loss
            
            # 循环终止条件:
            # 1.迭代次数达到指定的最大次数
            # 2.当前轮与上一轮的损失差小于指定的阀值
            # 3.误分类点个数为0
            if cur_iter >= self.max_iter or abs(loss_diff) < self.loss_tolerance or wrong_num == 0:
                break;
    
    # 预测
    def predict(self, x):
        y_pred = np.dot(self.w, x) + self.b
        return 1 if y_pred >= 0 else -1
                


# 测试
def main():
    
    X, y = load_iris(return_X_y=True)
    y[:50] = -1
    x_train, x_test, y_train, y_test = train_test_split(X[:100], y[:100], train_size=0.7, shuffle=True)
        
    model = Perceptron()
    model.train(x_train, y_train)
        
    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(x_test[i])
        if y_pred == y_test[i]:
            n_right += 1
        else:
            print("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(y_test[i], y_pred))
    print("Perceptron模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))
    print(n_right, n_test)
    
    
    
if __name__ == "__main__":
    main()
