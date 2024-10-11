# from datetime import time
import datetime
import math
import time
import pandas as pd
import numpy as np

def load_data() :

    # 读取文件
    df = pd.read_excel('data1.xlsx',index_col=0)

    # 特征工程

    # 第一步
    '''
     删除“echoBuffer”、“merchantCity”、“marchantState”、
     “merchantZip”、“posOnPremises”、“recurringAuthInd”，
    因为它们100%丢失。
    '''
    column_del  = ['echoBuffer','merchantCity','merchantState','merchantName',
    'merchantZip','posOnPremises','recurringAuthInd']
    df.drop(column_del, axis=1, inplace=True)


    # 第二步 customerId , accountNumber 重复，保留customerId
    df.drop(['accountNumber'],axis=1,inplace=True)


    # 第三步
    # 创建一个名为“limit_usage”的新列，以显示信用额度的使用百分比
    #（creditLimit-availableMoney）/creditLimit
    m, n = df.shape
    df_values = df.values
    # print(df_values)
    limit_usage_list = []
    for i in range(m) :
        creditLimit = float(df_values[i,1])
        availableMoney = float(df_values[i,2])
        limit_usage = (creditLimit - availableMoney)/ creditLimit
        limit_usage_list.append(limit_usage)
    # 插入“limit_usage”的新列
    df.insert(loc=3 , column='limit_usage', value=limit_usage_list)


    # 第四步
    # 将object 类型 转化成分类
    df['creditLimit'] = pd.Categorical(df['creditLimit'],df['creditLimit'].unique())
    # print(df['creditLimit'][0] )

    # 第五步
    # 删除acqCountry并保留merchantCountryCode
    # 创建了一个新列，以指示这两列是否相同。
    lsCountry = list(df['acqCountry']==df['merchantCountryCode'] )
    df.drop(['acqCountry'], axis=1, inplace=True)
    df.insert(loc=8, column='isSame', value=lsCountry)

    # 第六步
    # 创建了另一列“CVV_not_match”，以显示这两列是否不同
    lscard = list(df['cardCVV'] != df['enteredCVV'])
    df.insert(loc=15, column='CVV_not_match', value=lscard)

    # 第七步
    # 删除cardCVV  ，enteredCVV
    df.drop(['cardCVV','enteredCVV'],axis=1, inplace=True)
    # 第八步
    new_columnname = 'expiration_in_year'
    # 创建一个新列“expiration_in_year”以计算到期前还剩多少年
    # 由于数据仅包含月和年，请将“currentExpDate”转换为日期格式，并将“15
    # 日”指定为每月的估计日期。
    mp = dict()
    for index , c in enumerate(df.columns ):
        mp[c] = index
    for i in range(m) :
        currentExpDate = df.iloc[i,mp['currentExpDate']]
        # 03/2029
        month = currentExpDate.split('/')[0]
        year = currentExpDate.split('/')[1]
        day = str(15)
        # 补全
        date_str = "{0}-{1:0>2s}-{2:0>2s}".format(year, month, day)
        nd = datetime.date(*map(int, date_str.split('-')))
        # 修改
        df.iat[i, mp['currentExpDate']] = nd

    df['currentExpDate'] = pd.to_datetime(df['currentExpDate'], format='%Y-%m-%d', errors='coerce')  # 转换
    # 开户日期
    df['accountOpenDate'] = pd.to_datetime(df['accountOpenDate'], format='%Y-%m-%d', errors='coerce')  # 转换
    # 创建一个新列“expiration_in_year”以计算到期前还剩多少年
    # currentExpDate - localtime
    nowtime = time.strftime('%Y-%m-%d', time.localtime())
    nt = pd.to_datetime(nowtime, format='%Y-%m-%d', errors='coerce')
    # 计算天数差
    p1 = ((df['currentExpDate']-nt)/pd.Timedelta(1, 'D')).fillna(0).astype(float)
    # 计算年差
    # print(p/365)
    df.insert(loc=19, column='expiration_in_year', value=p1/365)
    # 删除原始列
    df.drop(['currentExpDate'], axis=1, inplace=True)

    # 创建一个新的列“years_opened”，以计算账户已开立的年份
    p2 = ((nt-df['accountOpenDate'])/pd.Timedelta(1, 'D')).fillna(0).astype(float)
    df.insert(loc=20, column='years_opened', value=p2/365)

    # 第九步
    # 创建一个新列“Address_changed”，以指示客户在开户后是否更改了地址
    Address_changed = list(df['accountOpenDate'] == df['dateOfLastAddressChange'])
    df.insert(loc=21, column='Address_changed', value=Address_changed)
    # 第十步
    # TODO
    # 还没做完
    # df2 = df.groupby(['customerId'], axis=0).agg({"cardLast4Digits": "nunique"}).reset_index()
    # # print('df2 : ' , list(df2['cardLast4Digits'] ))
    # p = list(df2['cardLast4Digits'])
    #
    # df.insert(loc=26, column='n_transac_by_this_c1tomerID', value=p)


    # 第十一步
    # 将“transactionDateTime”转换为DateTime格式
    transactionDateTime = df['transactionDateTime']
    transaction_months  = []
    transaction_years = []
    transaction_days = []
    transaction_weekdays = []
    isworkday = []
    transactionDateTime_Hour = []
    for tdt in transactionDateTime :
        ttt = tdt.split('T')[1]
        tdt = tdt.split('T')[0]
        transaction_months.append(int(tdt.split('-')[1] ))
        transaction_years.append(int(tdt.split('-')[0]) )
        transaction_days.append(int(tdt.split('-')[2]))
        weekdays = datetime.date(int(tdt.split('-')[0]), int(tdt.split('-')[1]), int(tdt.split('-')[2] )).isoweekday()
        transaction_weekdays.append(weekdays)
        if weekdays in [1,2,3,4,5] :
            isworkday.append('workday')
        else:
            isworkday.append('Weekend')
        # 小时数
        hours_ = ttt.split(':')[0]

        transactionDateTime_Hour.append(hours_)

    df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'], format='%Y-%m-%d', errors='coerce')  # 转换
    # 创建“transactionDateTime_Month”列以计算交易时间的月份
    df.insert(loc=22, column='transactionDateTime_Month', value=transaction_months)
    # 创建“transactionDateTime_Weekdays”列以计算事务时间的星期几
    df.insert(loc=23, column='transactionDateTime_Weekdays', value=transaction_weekdays)
    # 创建“transactionDateTime_Workday”列以计算是工作日还是周末
    df.insert(loc=24, column='transactionDateTime_Workday', value=isworkday)
    # 创建“transactionDateTime_Hour”列以计算一天的事务时间中的小时数
    df.insert(loc=25, column='transactionDateTime_Hour', value=transactionDateTime_Hour)

    # 创建“transactionDateTime_absolute”列以计算事务时间的绝对结束时间
    # TODO
    # 还没做完

    # 删除原始列“transactionDateTime”
    df.drop(['transactionDateTime'], axis=1, inplace=True)
    df.drop(['accountOpenDate'], axis=1, inplace=True)
    df.drop(['dateOfLastAddressChange'], axis=1, inplace=True)

    # 12 将isFraud、expirationDateKeyInMatch、cardPresent转换为分类数据
    # 将object 类型 转化成分类
    df['isFraud'] = pd.Categorical(df['isFraud'], df['isFraud'].unique())
    df['expirationDateKeyInMatch'] = pd.Categorical(df['expirationDateKeyInMatch'], df['expirationDateKeyInMatch'].unique())
    df['cardPresent'] = pd.Categorical(df['cardPresent'], df['cardPresent'].unique())

    #13 transactionAmoun
    # 如前所述，对transactionAmount使用log处理
    df['transactionAmount'] = df['transactionAmount'].apply(np.log1p)
    #14 删除交易金额为0美元的记录
    df[df['transactionAmount']==0 ]
    df = df[(df.transactionAmount != 0) ]

    # 将字符转换成数值型

    merchantCountryCode_mapping = {'US': 1, 'CAN': 3,'MEX':4,'PR':2}
    df['merchantCountryCode'] = df['merchantCountryCode'].map(merchantCountryCode_mapping)

    merchantCategoryCode_mapping = {'airline':1,'auto':2,'cable/phone':3,'entertainment':4
                                    ,'fastfood':5,'food':6,
                                    'food_delivery':7,
                                    'fuel':8,
                                    'furniture':9,'gym':10,
                                    'health':11,'hotels':12,
                                    'mobileapps':13,'online_gifts':14,
                                    'online_retail':15,'online_subscriptions':16,
                                    'personal care':17,'rideshare':18,
                                    'subscriptions':19

                                    }
    df['merchantCategoryCode'] = df['merchantCategoryCode'].map(merchantCategoryCode_mapping)

    transactionType_mapping ={'PURCHASE':1,'REVERSAL':0}
    df['transactionType'] = df['transactionType'].map(transactionType_mapping)

    print(type(df['CVV_not_match'][0]) )
    CVV_not_match_mapping = {False:0,True:1}
    cardPresent_mapping =  {False:0,True:1}
    expirationDateKeyInMatch_mapping = {False:0,True:1}
    Address_changed_mapping = {False:0,True:1}
    isFraud_mapping = {False:0,True:1}
    transactionDateTime_Workday_mapping = {'Weekend':0,'workday':1}

    df['CVV_not_match'] = df['CVV_not_match'].map(CVV_not_match_mapping)
    df['cardPresent'] = df['cardPresent'].map(cardPresent_mapping)
    df['expirationDateKeyInMatch'] = df['expirationDateKeyInMatch'].map(expirationDateKeyInMatch_mapping)
    df['Address_changed'] = df['Address_changed'].map(Address_changed_mapping)
    df['isFraud'] = df['isFraud'].map(isFraud_mapping)
    df['transactionDateTime_Workday'] = df['transactionDateTime_Workday'].map(transactionDateTime_Workday_mapping)

    df.drop(['customerId'], axis=1, inplace=True)
    df.drop(['isSame'], axis=1, inplace=True)
    print(df.columns)
    print(len(df.columns))

    df.to_excel('AfterfeatureEngineering_file.xlsx')




if __name__ == '__main__':

    load_data()
