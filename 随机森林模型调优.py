from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    f1_score

from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate

x = None
y = None
# 目标函数
def rf_cv(n_estimators, min_samples_split, max_features, max_depth,min_samples_leaf):
    # 输出为模型交叉验证10次f1-score均值，作为我们的目标函数
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_features=min(max_features, 0.999),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=2)

    # score = cross_val_score(rf, x, y, cv=10).mean()
    # scoring = {'accuracy': make_scorer(accuracy_score),
    #            'precision': make_scorer(precision_score, average='micro'),
    #            'recall': make_scorer(recall_score, average='micro'),
    #            'f1_score': make_scorer(f1_score, average='micro')}
    #10折交叉验证
    scores = cross_val_score(rf, x, y,cv=10, scoring= make_scorer(f1_score, average = 'micro') ).mean()


    return scores


def load_data(filename):
    df = pd.read_excel(filename, index_col=0 ,header=0)
    # drop time and name , those str type
    df = df.dropna()
    col = len(df.columns)
    data_X = df.iloc[:, :col-1]
    data_Y = df.iloc[:, -1]

    return data_X,data_Y

if __name__ =='__main__':


    # 数据文件
    filename = 'AfterfeatureEngineering_file.xlsx'
    # 加载数据集
    data_X, data_y = load_data(filename=filename)

    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=123,
                                                        train_size=0.8, shuffle=True)
    x = x_train.values
    y = y_train.values.ravel()

    # 贝叶斯优化
    rf_bo = BayesianOptimization(rf_cv,
            {
                'n_estimators': (10, 300),
                'min_samples_split': (2, 25),
                'max_features': (10, 30),
                'max_depth': (5, 20),
                'min_samples_leaf':(1,20)
             }
        )
    rf_bo.maximize(init_points=20, n_iter=100)# 抽取多少个初始观测值/总共观测/迭代次数
    print(rf_bo.res)

