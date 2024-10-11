
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(filename):
    df = pd.read_excel(filename, index_col=0 ,header=0)
    # drop time and name , those str type
    # print(df)

    # 去掉nan数据样本
    df = df.dropna()
    col = len(df.columns)
    data_X = df.iloc[:, :col - 1]
    data_Y = df.iloc[:, -1]
    return data_X,data_Y

if __name__ == '__main__':
    # 数据文件
    # filename = 'bankdata.xlsx'
    filename = 'AfterfeatureEngineering_file.xlsx'
    # 加载数据集
    data_X, data_y = load_data(filename=filename)
    print(data_X)
    print(data_y)
    is_Fraud_Fasle = []
    is_Fraud_True = []
    for i in data_y :
        if i ==1:
            is_Fraud_True.append(i)
        else:
            is_Fraud_Fasle.append(i)
    total_data = len(data_y)
    Fasle_Fraud_per = float(len(is_Fraud_Fasle)/total_data)
    True_Fraud_per = float(len(is_Fraud_True)/total_data)
    print(Fasle_Fraud_per)
    print(True_Fraud_per)
    x_Data = ['Fasle_Fraud','True_Fraud']
    y_Data = [Fasle_Fraud_per,True_Fraud_per]

    for i in range(len(x_Data)):
        plt.bar(x_Data[i], y_Data[i])
    label = ['Fasle_Fraud', 'True_Fraud']
    plt.legend(label)
    plt.ylabel('percentage(%)')
    plt.show()
