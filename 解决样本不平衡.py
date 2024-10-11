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

# 安装 imbalanced
# pip install imbalanced-learn -i https://pypi.douban.com/simple

def load_data(filename):
    df = pd.read_excel(filename, index_col=0 ,header=0)

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

def init_excel(df_performance):
    # df_preformance = pd.DataFrame()
    df_performance['Name'] = 'ddd'
    df_performance['Train_AUC'] = 0
    df_performance['Train_Precision'] = 0
    df_performance['Train_Accuracy'] = 0
    df_performance['Train_Recall'] = 0
    df_performance['Train_f1-score'] = 0
    df_performance['Train_support'] = 0

    df_performance['Test_AUC'] = 0
    df_performance['Test_Precision'] = 0
    df_performance['Test_Accuracy'] = 0
    df_performance['Test_Recall'] = 0
    df_performance['Test_f1-score'] = 0
    df_performance['Test_support'] = 0

    return df_performance

def plot_ROC_curve(fp,tp ,model_name,medthod_name):
    # 画出 ROC 曲线
    plt.plot(fp, tp, lw=1.5, label='{} (AUC={:.3f})'.format(model_name+medthod_name, auc(fp, tp)),
             color='orange')
    # plt.plot(fpr_xg, tpr_xg, lw=1.5, label='{} (AUC={:.3f})'.format('XGBoost', auc(fpr_xg, tpr_xg)), color='green')
    # plt.plot(fpr_gb, tpr_gb, lw=1.5, label='{} (AUC={:.3f})'.format('GradientBoost', auc(fpr_gb, tpr_gb)), color='blue')
    plt.plot([0, 1], [0, 1], '--', lw=1.5, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC Curve', fontsize=10)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right', fontsize=6)
    plt.savefig(model_name+medthod_name+'ROC_curve.png', dpi=512)
    plt.show()
# 解决样本不均衡
def fun(x_train, x_test, y_train, y_test,model,method,method_name,df_preformance,model_name) :

    print('======'+method_name+'=======')
    x_resample, y_resample = method.fit_resample(x_train, y_train.ravel())
    # print('重采样x_train', len(x_resample))
    print('sampling data size {}'.format(len(y_resample)))
    # 模型训练
    model.fit(x_resample, y_resample)
    # 计算 ROC 的 TP,FP
    # 训练集的
    fpr_rf_Train, tpr_rf_Train, thresholds_rf_Train = roc_curve(y_resample, model.predict_proba(x_resample)[:, 1])
    # 测试集的
    fpr_rf_Test, tpr_rf_Test, thresholds_rf_Test = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

    # Performance on Train dataset
    # Train
    # AUC ,Accuracy
    predictions_train = model.predict(x_resample)
    train_auc_ = auc(fpr_rf_Train, tpr_rf_Train)
    train_acc = accuracy_score(y_resample, predictions_train)
    train_pre, train_rec, train_f1, train_sup = precision_recall_fscore_support(y_resample, predictions_train)
    # print(classification_report(y_resample, predictions_train))
    # print('AUC: {:.3f}'.format(train_auc_))
    # print("precision:", train_pre[0], "\nrecall:", train_rec[0], "\nf1-score:", train_f1[0], "\nsupport:", train_sup[0])


    # Performance on Test dataset
    # Test
    # AUC ,Accuracy
    predictions_test = model.predict(x_test)
    test_auc_ = auc(fpr_rf_Test, tpr_rf_Test)
    test_acc = accuracy_score(y_test, predictions_test)
    test_pre, test_rec, test_f1, test_sup = precision_recall_fscore_support(y_test, predictions_test)
    # print(classification_report(y_test, predictions_test))
    print('test dataset : ')
    print('AUC: {:.3f}'.format(test_auc_))
    print("precision:", test_pre[0], "\nrecall:", test_rec[0], "\nf1-score:", test_f1[0], "\nsupport:", test_sup[0])

    # 画出混淆矩阵
    cm_rf = confusion_matrix(y_test, predictions_test)
    cm_play = ConfusionMatrixDisplay(cm_rf).plot()
    plt.title(model_name+':'+'confusion_matrix')
    plt.savefig(model_name+'_'+'confusion_matrix.png')
    plt.close()

    # SMOTE采样性能
    df_preformance = df_preformance.append(
        {'Name':method_name,'Train_AUC':train_auc_,'Train_Precision':train_pre[0],'Train_Accuracy':train_acc,
            'Train_Recall':train_rec[0],'Train_f1-score':train_f1[0],'Train_support':train_sup[0],
         'Test_AUC':test_auc_,'Test_Precision':test_pre[0],'Test_Accuracy':test_acc,
         'Test_Recall':test_rec[0],'Test_f1-score':test_f1[0],'Test_support':test_sup[0]

         },
        ignore_index=True)

    return df_preformance,(fpr_rf_Test,tpr_rf_Test)

def Performance_ON_differentModel(model ,df_performance,file_name,model_name) :
    '''

    :param model: 模型 比如 RandomForestClassifer
    :param df_performance: DataFrame文件对象
    :param file_name: DataFrame文件名字
    :param model_name: 模型名字
    :return:
    '''
    # print('over-sampling data size {}'.format(len(data_y)))

    # over-sampling
    smo = SMOTE(random_state=0, sampling_strategy='auto',
                k_neighbors=5,
                n_jobs=1)
    df_performance,fp_tp1 = fun(x_train, x_test, y_train, y_test, model, smo, 'SMTO', df_performance,model_name)


    sblsmo = BorderlineSMOTE(random_state=0, kind="borderline-2"
                             , sampling_strategy='auto',
                             k_neighbors=5,
                             n_jobs=1, m_neighbors=10,
                             )
    df_performance ,fp_tp2= fun(x_train, x_test, y_train, y_test, model, sblsmo, 'BorderlineSMOTE', df_performance,model_name)

    # svmsmo = SVMSMOTE(
    #     sampling_strategy='auto',random_state=0,k_neighbors=5, n_jobs=1,
    #     m_neighbors=10,svm_estimator=None,
    #     out_step=0.5)
    # df_preformance = fun(x_train, x_test, y_train, y_test, rf, svmsmo, 'SVMSMOTE', df_preformance,model_name)

    ada = ADASYN(random_state=0, sampling_strategy='auto',
                 n_neighbors=5, n_jobs=1,
                 )
    df_performance ,fp_tp3= fun(x_train, x_test, y_train, y_test, model, ada, 'ADASYN', df_performance,model_name)
    # under-sampling
    model_undersample = RandomUnderSampler(random_state=0)
    df_performance ,fp_tp4= fun(x_train, x_test, y_train, y_test, model, model_undersample, 'RandomUnder', df_performance,model_name)

    nm1 = NearMiss(version=1)
    df_performance ,fp_tp5 = fun(x_train, x_test, y_train, y_test, model, nm1, 'NearMiss', df_performance,model_name)

    ls = []
    ls.append(fp_tp1)
    ls.append(fp_tp2)
    ls.append(fp_tp3)
    ls.append(fp_tp4)
    ls.append(fp_tp5)


    df_performance.to_excel(file_name)
    return df_performance,ls
if __name__ == '__main__':

    # 数据文件
    filename = 'AfterfeatureEngineering_file.xlsx'
    # 加载数据集
    data_X, data_y = load_data(filename=filename)
    print('origin data size {}'.format( len(data_y)) )
    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=123,
                                                     train_size=0.8,shuffle=True)
    print('train data size {}'.format(len(y_train)))
    # 随机森林分类器
    rf = RandomForestClassifier(random_state=0,n_estimators=40,max_depth=7,max_features=0.26,
                                min_samples_leaf=18,min_samples_split=23,
                                )
    # XGboost分类器
    xg = XGBClassifier(random_state=0, use_label_encoder=False, n_estimators=263, max_depth=8)
    # GradientBoosting 分类器
    gb = GradientBoostingClassifier(random_state=0, n_estimators=488, max_depth=5,
                                    max_features=0.4, min_samples_split=2, min_samples_leaf=10)

    #初始化excel文件
    df_performance1 = pd.DataFrame()
    df_performance1 = init_excel(df_performance1)
    rf_performance = 'Performance_rf.xlsx'
    # 在随机森林模型上分别使用SMOTE,borderlineSMOTE,(ADASYN)自适应合成上采样 ，三种上over-sampling方式
    # 以及RandomUnderSampler，NearMiss 两种下采样 under-sampling
    df_performance1,l1 = Performance_ON_differentModel(rf,df_performance1,rf_performance,'randomforest')



    # 初始化excel文件
    df_performance2 = pd.DataFrame()
    df_preformance2 = init_excel(df_performance2)
    xg_performance = 'Performance_xg.xlsx'
    # 在XGBoost模型上分别使用SMOTE,borderlineSMOTE,(ADASYN)自适应合成上采样 ，三种上over-sampling方式
    # 以及RandomUnderSampler，NearMiss 两种下采样 under-sampling
    df_performance2 ,l2= Performance_ON_differentModel(xg,df_performance2,xg_performance,'XBGboost')

    # 初始化excel文件
    df_performance3 = pd.DataFrame()
    df_performance3 = init_excel(df_performance3)
    gb_performance = 'Performance_gb.xlsx'
    # 在GradientBoosting模型上分别使用SMOTE,borderlineSMOTE,(ADASYN)自适应合成上采样 ，三种上over-sampling方式
    # 以及RandomUnderSampler，NearMiss 两种下采样 under-sampling
    # 并将Performance写在excel中
    df_performance3 ,l3= Performance_ON_differentModel(gb, df_performance3, gb_performance,'GradientBoosting')



    # 分别画出，5中不同的解决样本不平衡方法下，3个模型在测试集上面的ROC曲线

    sampling_name= ['SMTO','BorderlineSMOTE','ADASYN','RandomUnder','NearMiss']
    for i in range(5) :
        l1_fp , l1_tp = l1[i]
        l2_fp ,l2_tp = l2[i]
        l3_fp,l3_tp  = l3[i]
        plt.plot(l1_fp, l1_tp, lw=1.5, label='{} (AUC={:.3f})'.format('RandomForest', auc(l1_fp, l1_tp)),
                 color='orange')
        plt.plot(l2_fp, l2_tp, lw=1.5, label='{} (AUC={:.3f})'.format('XGBoost', auc(l2_fp, l2_tp)), color='green')
        plt.plot(l3_fp, l3_tp, lw=1.5, label='{} (AUC={:.3f})'.format('GradientBoost', auc(l3_fp, l3_tp)), color='blue')
        plt.plot([0, 1], [0, 1], '--', lw=1.5, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(sampling_name[i]+'-'+'ROC Curve', fontsize=10)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc='lower right', fontsize=6)
        plt.savefig( sampling_name[i]+'ROC_curve.png', dpi=512)
        plt.show()


