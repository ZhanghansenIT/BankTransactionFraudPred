from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

x = None
y = None
# 目标函数
def xg_cv(n_estimators,  max_depth):

    # ensemble.GradientBoostingRegressor
    xg = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),

        random_state=2,
        use_label_encoder=False
    )
    scores = cross_val_score(xg, x, y, cv=10, scoring=make_scorer(f1_score, average='micro')).mean()
    # score = cross_val_score(xg, x, y, cv=10).mean()
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
    xg_bo = BayesianOptimization(xg_cv,
            {'n_estimators': (10, 300),

            'max_depth': (5, 20),
             }
        )
    xg_bo.maximize(init_points=20, n_iter=200)# 抽取多少个初始观测值/总共观测/迭代次数
    print(xg_bo.res)

