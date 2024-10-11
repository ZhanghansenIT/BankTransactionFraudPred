import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, explained_variance_score, roc_curve, confusion_matrix, accuracy_score, \
    f1_score, mean_squared_error, ConfusionMatrixDisplay, auc, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier  # 导入随机森林
import time, datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss



def load_data(filename):
    df = pd.read_excel(filename, index_col=0 ,header=0)
    # drop time and name , those str type

    transactionDateTime_Workday_mapping = {'Weekend': 0, 'workday': 1}
    transactionDateTime_Month_mapping = {'January':1,'February':2,
                                         'March':3,'A2il':4,'May':5,
                                         'June':6,'July':7,'August':8,
                                         'September':9,'October':10,
                                         'November':11,'December':12}

    transactionDateTime_Weekdays_mapping = {'Monday':1,'Tuesday':2,'Wednesday':3,
                                    'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    df['transactionDateTime_Month'] = df['transactionDateTime_Month'].map(transactionDateTime_Month_mapping)

    df['transactionDateTime_Weekdays'] = df['transactionDateTime_Weekdays'].map(
        transactionDateTime_Weekdays_mapping)
    df['transactionDateTime_Workday'] = df['transactionDateTime_Workday'].map(
        transactionDateTime_Workday_mapping)
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


def model_performance() :
    # performance
    # AUC
    # Precision
    # Recall
    pass

# 解决样本不均衡
def fun(x_train, x_test, y_train, y_test,model,method,method_name,df_preformance) :
    print('原始x_train', len(x_train))
    x_resample, y_resample = method.fit_resample(x_train, y_train.astype('int'))
    print('重采样x_train', len(x_resample))
    model.fit(x_resample, y_resample)

    fpr_rf_Train, tpr_rf_Train, thresholds_rf_Train = roc_curve(y_resample, model.predict_proba(x_resample)[:, 1])

    # 计算 ROC 的 TP,FP
    # 随机森林的
    fpr_rf_Test, tpr_rf_Test, thresholds_rf_Test = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

    # Performance on Train dataset
    # Train
    # AUC ,Accuracy
    predictions_train = model.predict(x_resample)
    train_auc_ = auc(fpr_rf_Train, tpr_rf_Train)
    train_acc = accuracy_score(y_resample, predictions_train)
    train_pre, train_rec, train_f1, train_sup = precision_recall_fscore_support(y_resample, predictions_train)

    print('AUC: {:.3f}'.format(train_auc_))
    print("precision:", train_pre[0], "\nrecall:", train_rec[0], "\nf1-score:", train_f1[0], "\nsupport:", train_sup[0])

    # Performance on Test dataset
    # Test
    # AUC ,Accuracy

    predictions_test = model.predict(x_test)
    test_auc_ = auc(fpr_rf_Test, tpr_rf_Test)
    test_acc = accuracy_score(y_resample, predictions_train)
    test_pre, test_rec, test_f1, test_sup = precision_recall_fscore_support(y_test, predictions_test)

    # SMOTE采样性能
    df_preformance = df_preformance.append(
        {'Name':method_name,'Train_AUC':train_auc_,'Train_Precision':train_pre[0],'Train_Accuracy':train_acc,
            'Train_Recall':train_rec[0],'Train_f1-score':train_f1[0],'Train_support':train_sup[0],
         'Test_AUC':test_auc_,'Test_Precision':test_pre[0],'Test_Accuracy':test_acc,
         'Test_Recall':test_rec[0],'Test_f1-score':test_f1[0],'Test_support':test_sup[0]

         },
        ignore_index=True)

    # print(df_preformance)
    return df_preformance
if __name__ == '__main__':

    # 数据文件
    filename = 'bankdata.xlsx'
    # 加载数据集
    data_X, data_y = load_data(filename=filename)
    # print(data_y.columns)

    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=123,
                                                     train_size=0.8,shuffle=True)

    # 随机森林分类器
    rf = RandomForestClassifier(random_state=0,n_estimators=40,max_depth=7,max_features=0.26,
                                min_samples_leaf=18,min_samples_split=23,

                                )

    # 设置随机种子控制
    # SMOTE采样
    # Synthetic Minority Oversampling Technique

    # print('原始x_train',len(x_train))
    # x_resample, y_resample = smo.fit_resample(x_train, y_train)
    # print('重采样',len(x_resample))
    # rf.fit(x_resample, y_resample)

    df_preformance = pd.DataFrame()
    df_preformance['Name'] = 'ddd'
    df_preformance['Train_AUC'] = 0
    df_preformance['Train_Precision'] = 0
    df_preformance['Train_Accuracy'] = 0
    df_preformance['Train_Recall'] = 0
    df_preformance['Train_f1-score'] = 0
    df_preformance['Train_support'] = 0

    df_preformance['Test_AUC'] = 0
    df_preformance['Test_Precision'] = 0
    df_preformance['Test_Accuracy'] = 0
    df_preformance['Test_Recall'] = 0
    df_preformance['Test_f1-score'] = 0
    df_preformance['Test_support'] = 0


    smo = SMOTE(random_state=0,sampling_strategy='auto',
            k_neighbors=5,
            n_jobs=1)
    df_preformance = fun(x_train, x_test, y_train, y_test,rf,smo,'SMTO',df_preformance)

    sblsmo = BorderlineSMOTE(random_state=0, kind="borderline-1"
                             ,sampling_strategy='auto',
     k_neighbors=5,
     n_jobs=1, m_neighbors=10,
     )
    df_preformance = fun(x_train, x_test, y_train, y_test, rf, sblsmo, 'BorderlineSMOTE', df_preformance)


    # svmsmo = SVMSMOTE(
    #     sampling_strategy='auto',random_state=0,k_neighbors=5, n_jobs=1,
    #     m_neighbors=10,svm_estimator=None,
    #     out_step=0.5)
    # df_preformance = fun(x_train, x_test, y_train, y_test, rf, svmsmo, 'SVMSMOTE', df_preformance)

    ada = ADASYN(random_state=0,sampling_strategy='auto',
      n_neighbors=5, n_jobs=1,
      )
    df_preformance = fun(x_train, x_test, y_train, y_test, rf, ada, 'ADASYN', df_preformance)



    # # 查看数据是否不平衡
    # is_fraud_true = []
    # is_fraud_false = []
    #
    # for d in y_resample :
    #     if d == 1 :
    #         is_fraud_true.append(1)
    #     else:
    #         is_fraud_false.append(0)
    #
    # print(f'is_fraud_true : {len(is_fraud_true)}' )
    # print(f'is_fraud_false : {len(is_fraud_false)}')

    #
    #
    #
    # # 测试
    # predictions_rf = rf.predict(x_test)
    # # 混淆矩阵
    # cm_rf = confusion_matrix(y_test, predictions_rf)
    # ascore_rf = accuracy_score(y_test, predictions_rf)
    # features = data_X.columns
    # importances_rf = rf.feature_importances_
    # plot_featureImportence(features, importances_rf, 'RandomForest', 'green')
    #
    #
    # # 模型评估
    # print('Calculation model evaluation RF model : ')
    # print(f'RF_accuaracy score : {ascore_rf} ')
    # cal_model_evaluation(y_test, predictions_rf)
    # #
    #
    #
    # # 画出混淆矩阵
    # plt.figure(1)
    # cm1_play = ConfusionMatrixDisplay(cm_rf).plot()
    # plt.title('confusion_matrix_RF')
    # plt.savefig('confusion_matrix_RF.png')
    # plt.close()
    #
    # fpr_rf_Train, tpr_rf_Train, thresholds_rf_Train = roc_curve(y_resample, rf.predict_proba(x_resample)[:,1])
    #
    # # 计算 ROC 的 TP,FP
    # # 随机森林的
    # fpr_rf_Test, tpr_rf_Test, thresholds_rf_Test = roc_curve(y_test, rf.predict_proba(x_test)[:,1])


    # roc_auc_rf = metrics.auc(fpr_rf_Test, tpr_rf_Test)
    # # 画出 ROC 曲线
    # plt.plot(fpr_rf_Test, tpr_rf_Test, lw=1.5, label='{} (AUC={:.3f})'.format('RandomForest', auc(fpr_rf_Test, tpr_rf_Test)), color='orange')
    # # plt.plot(fpr_xg, tpr_xg, lw=1.5, label='{} (AUC={:.3f})'.format('XGBoost', auc(fpr_xg, tpr_xg)), color='green')
    # # plt.plot(fpr_gb, tpr_gb, lw=1.5, label='{} (AUC={:.3f})'.format('GradientBoost', auc(fpr_gb, tpr_gb)), color='blue')
    # plt.plot([0, 1], [0, 1], '--', lw=1.5, color='grey')
    # plt.axis('square')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.title('ROC Curve', fontsize=10)
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc='lower right',fontsize=6)
    # plt.savefig('ROC_curve.png',dpi=512)
    # plt.show()


    # Performance on Train dataset
    # Train
    # AUC ,Accuracy
    # predictions_train = rf.predict(x_resample)
    # train_auc_ = auc(fpr_rf_Train, tpr_rf_Train)
    # train_acc = accuracy_score(y_resample,predictions_train)
    # train_pre, train_rec, train_f1, train_sup = precision_recall_fscore_support(y_resample, predictions_train)
    #
    # print('AUC: {:.3f}'.format(train_auc_))
    # print("precision:", train_pre[0], "\nrecall:", train_rec[0], "\nf1-score:", train_f1[0], "\nsupport:", train_sup[0])

    # Performance on Test dataset
    # Test
    # AUC ,Accuracy

    # predictions_test = rf.predict(x_test)
    # test_auc_ = auc(fpr_rf_Test, tpr_rf_Test)
    # test_acc = accuracy_score(y_resample, predictions_train)
    # test_pre, test_rec, test_f1, test_sup = precision_recall_fscore_support(y_test, predictions_test)

    # columns =  ['name','AUC', 'Precision', 'Accuracy','Recall', 'f1-score', 'support']
    # dic1 = {'Name':'','AUC':0,'Precision':0,'Accuracy':0,
    #         'Recall':0,'f1-score':0,'support':0}
    #

    # df_preformance = pd.DataFrame()
    # df_preformance['Name'] = 'ddd'
    # df_preformance['Train_AUC'] = 0
    # df_preformance['Train_Precision'] = 0
    # df_preformance['Train_Accuracy'] = 0
    # df_preformance['Train_Recall'] = 0
    # df_preformance['Train_f1-score'] = 0
    # df_preformance['Train_support'] = 0
    #
    # df_preformance['Test_AUC'] = 0
    # df_preformance['Test_Precision'] = 0
    # df_preformance['Test_Accuracy'] = 0
    # df_preformance['Test_Recall'] = 0
    # df_preformance['Test_f1-score'] = 0
    # df_preformance['Test_support'] = 0




    # SMOTE采样
    # df_preformance = df_preformance.append(
    #     {'Name':'SMTOE','Train_AUC':train_auc_,'Train_Precision':train_pre[0],'Train_Accuracy':train_acc,
    #         'Train_Recall':train_rec[0],'Train_f1-score':train_f1[0],'Train_support':train_sup[0],
    #      'Test_AUC':test_auc_,'Test_Precision':test_pre[0],'Test_Accuracy':test_acc,
    #      'Test_Recall':test_rec[0],'Test_f1-score':test_f1[0],'Test_support':test_sup[0]
    #
    #      },
    #     ignore_index=True)
    #
    # # downsampling


    #
    # # 使用欠采样
    # print("使用欠采样====")
    #
    #

    model_undersample = RandomUnderSampler(random_state=0)
    df_preformance = fun(x_train, x_test, y_train, y_test, rf, model_undersample, '随机采样', df_preformance)

    nm1 = NearMiss(version=1)
    df_preformance = fun(x_train, x_test, y_train, y_test, rf, nm1,'NearMiss' ,df_preformance)



    # x_undersample_resampled, y_undersample_resampled = nm1.fit_resample(x_train, y_train)
    # model_undersample = RandomUnderSampler()
    # #
    # # # .fit_sample(x_train, y_train)
    # # print(len(y_train))
    # # x_undersample_resampled, y_undersample_resampled = model_undersample.fit_resample(x_train,y_train)
    # print(len(y_undersample_resampled))
    #
    #
    # # x_undersample_resampled, y_undersample_resampled = RandomUnderSample(x_train,y_train)
    # rf.fit(x_undersample_resampled, y_undersample_resampled)
    #
    # fpr_rf_Train, tpr_rf_Train, thresholds_rf_Train = roc_curve(y_undersample_resampled, rf.predict_proba(x_undersample_resampled)[:, 1])
    #
    # # 计算 ROC 的 TP,FP
    # # 随机森林的
    # fpr_rf_Test, tpr_rf_Test, thresholds_rf_Test = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])
    #
    # # Performance on Train dataset
    # # Train
    # # AUC ,Accuracy
    # predictions_train = rf.predict(x_undersample_resampled)
    # train_auc_ = auc(fpr_rf_Train, tpr_rf_Train)
    # train_acc = accuracy_score(y_undersample_resampled, predictions_train)
    # train_pre, train_rec, train_f1, train_sup = precision_recall_fscore_support(y_undersample_resampled, predictions_train)
    #
    # # Performance on Test dataset
    # # Test
    # # AUC ,Accuracy
    #
    # predictions_test = rf.predict(x_test)
    # test_auc_ = auc(fpr_rf_Test, tpr_rf_Test)
    # test_acc = accuracy_score(y_undersample_resampled, predictions_train)
    # test_pre, test_rec, test_f1, test_sup = precision_recall_fscore_support(y_test, predictions_test)
    #
    # df_preformance = df_preformance.append(
    #     {'Name': 'UNder-sampling', 'Train_AUC': train_auc_, 'Train_Precision': train_pre[0], 'Train_Accuracy': train_acc,
    #      'Train_Recall': train_rec[0], 'Train_f1-score': train_f1[0], 'Train_support': train_sup[0],
    #      'Test_AUC': test_auc_, 'Test_Precision': test_pre[0], 'Test_Accuracy': test_acc,
    #      'Test_Recall': test_rec[0], 'Test_f1-score': test_f1[0], 'Test_support': test_sup[0]
    #
    #      },
    #     ignore_index=True)
    df_preformance.to_excel('Performance4.xlsx')
    #
