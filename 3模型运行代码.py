import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, explained_variance_score, roc_curve, confusion_matrix, accuracy_score, \
    f1_score, mean_squared_error, ConfusionMatrixDisplay, auc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier  # 导入随机森林
import time, datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


def load_data(filename):
    df = pd.read_excel(filename, index_col=0 ,header=0)
    # drop time and name , those str type

    # transactionDateTime_Workday_mapping = {'Weekend': 0, 'workday': 1}
    # transactionDateTime_Month_mapping = {'January':1,'February':2,
    #                                      'March':3,'A2il':4,'May':5,
    #                                      'June':6,'July':7,'August':8,
    #                                      'September':9,'October':10,
    #                                      'November':11,'December':12}
    #
    # transactionDateTime_Weekdays_mapping = {'Monday':1,'Tuesday':2,'Wednesday':3,
    #                                 'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    #
    # df['transactionDateTime_Month'] = df['transactionDateTime_Month'].map(transactionDateTime_Month_mapping)
    #
    # df['transactionDateTime_Weekdays'] = df['transactionDateTime_Weekdays'].map(
    #     transactionDateTime_Weekdays_mapping)
    # df['transactionDateTime_Workday'] = df['transactionDateTime_Workday'].map(
    #     transactionDateTime_Workday_mapping)
    #
    # position1 = df.isnull().stack()[lambda x: x].index.tolist()
    # print(position1)
    # print(df['transactionDateTime_Workday'])

    # 去掉nan数据样本
    df = df.dropna()

    col = len(df.columns)
    data_X = df.iloc[:, :col - 1]
    data_Y = df.iloc[:, -1]
    return data_X,data_Y
def plot_featureImportence(features,importences :list,name,color ) :

    plt.figure(figsize=(10,6))
    indices = np.argsort(importences)
    plt.title(name +':importmence of Feature ',fontsize =10)
    plt.barh(range(len(indices)), importences[indices], color=color, align='center')
    plt.yticks(range(len(indices)), [features[i] for i in reversed(indices)], fontsize=6)
    plt.xlabel('Relative Importance')
    plt.ylabel('features',fontsize=6)
    plt.savefig('feature_importence'+name+'.png', dpi=512)
    plt.close()

#
def cal_model_evaluation(y_test,predictions) :


    F1_score = f1_score(y_test,predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("F1_score : " ,F1_score)
    print("MAE：", mae)
    print("MSE：", mse)
    print("RMSE: ", rmse)
if __name__ == '__main__':
    # 数据文件
    filename = 'AfterfeatureEngineering_file.xlsx'
    # 加载数据集
    data_X, data_y = load_data(filename=filename)
    print(data_X)
    print(np.shape(data_X))
    # print(data_y.columns)

    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=123,
                                                     train_size=0.8,shuffle=True)

    # 随机森林分类器
    rf = RandomForestClassifier(random_state=0,n_estimators=40,max_depth=13,max_features=18,
                                min_samples_leaf=18,min_samples_split=23,

                                )
    rf.fit(x_train, y_train)

    # 测试
    predictions_rf = rf.predict(x_test)
    # 混淆矩阵
    cm_rf = confusion_matrix(y_test, predictions_rf)
    ascore_rf = accuracy_score(y_test, predictions_rf)
    features = data_X.columns
    importances_rf = rf.feature_importances_
    plot_featureImportence(features, importances_rf, 'RandomForest', 'green')
    # 模型评估
    print('Calculation model evaluation RF model : ')
    print(f'RF_accuaracy score : {ascore_rf} ')
    cal_model_evaluation(y_test, predictions_rf)
    # 画出混淆矩阵
    plt.figure(1)
    cm1_play = ConfusionMatrixDisplay(cm_rf).plot()
    plt.title('confusion_matrix_RF')
    plt.savefig('confusion_matrix_RF.png')
    plt.close()



    # XGBoost分类器
    xg = XGBClassifier(random_state=0, use_label_encoder=False, n_estimators=263, max_depth=8)
    xg.fit(x_train, y_train)
    predictions_xg = xg.predict(x_test)
    cm_xg = confusion_matrix(y_test, predictions_xg)
    ascore_xg = accuracy_score(y_test, predictions_xg)
    importances_xg = xg.feature_importances_

    plot_featureImportence(features,importances_xg, 'XGBoost', 'orange')
    print('Calculation model evaluation Xgboost model : ')
    print(f'XG_accuaracy score : {ascore_xg} ')
    # print(y_test ,predictions_xg)
    cal_model_evaluation(y_test,predictions_xg)
    plt.figure(2)
    cm2_play = ConfusionMatrixDisplay(cm_xg).plot()
    plt.title('confusion_matrix_Xgboost')
    plt.savefig('confusion_matrix_XG.png')

    plt.close()


    # GradientBoosting模型
    gb = GradientBoostingClassifier(random_state=0, n_estimators=488, max_depth=5,
                                    max_features=0.4,min_samples_split=2,min_samples_leaf=10)
    gb.fit(x_train, y_train)
    predictions_gb = gb.predict(x_test)
    cm_gb = confusion_matrix(y_test, predictions_gb)
    ascore_gb = accuracy_score(y_test, predictions_gb)
    importances_gb = gb.feature_importances_

    plot_featureImportence(features, importances_gb, 'Gradientboost', 'blue')
    print('Calculation model evaluation Gradientboost model : ')
    print(f'XG_accuaracy score : {ascore_gb} ')
    cal_model_evaluation(y_test, predictions_gb)
    plt.figure(2)
    cm3_play = ConfusionMatrixDisplay(cm_gb).plot()
    plt.title('confusion_matrix_Gradientboost')
    plt.savefig('confusion_matrix_Gradientboost.png')

    plt.close()


    # 计算 ROC 的 TP,FP
    # 随机森林的
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
    # XGBoost的
    fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, xg.predict_proba(x_test)[:, 1])
    fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, gb.predict_proba(x_test)[:, 1])
    roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
    roc_auc_xg = metrics.auc(fpr_xg, tpr_xg)

    roc_auc_gb = metrics.auc(fpr_gb, tpr_gb)
    # plt.plot(fpr, tpr, label=roc_auc)

    # 画出 ROC 曲线
    plt.plot(fpr_rf, tpr_rf, lw=1.5, label='{} (AUC={:.3f})'.format('RandomForest', auc(fpr_rf, tpr_rf)), color='orange')
    plt.plot(fpr_xg, tpr_xg, lw=1.5, label='{} (AUC={:.3f})'.format('XGBoost', auc(fpr_xg, tpr_xg)), color='green')
    plt.plot(fpr_gb, tpr_gb, lw=1.5, label='{} (AUC={:.3f})'.format('GradientBoost', auc(fpr_gb, tpr_gb)), color='blue')
    plt.plot([0, 1], [0, 1], '--', lw=1.5, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC Curve', fontsize=10)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right',fontsize=6)
    plt.savefig('ROC_curve.png',dpi=512)
    plt.show()






