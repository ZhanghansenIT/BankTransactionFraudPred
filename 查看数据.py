
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    filename = '原始数据.xlsx'

    col_n = ['availableMoney', 'transactionAmount','isFraud']
    df = pd.read_excel(filename,index_col=0 ,header=0)

    isFraud_mapping = {False: 0, True: 1}


    df['isFraud'] = df['isFraud'].map(isFraud_mapping)
    data1 = pd.DataFrame(df, columns=col_n)
    print(data1)
    sns.pairplot(data1, hue='isFraud',height=2,diag_kind = 'hist')
    plt.show()

