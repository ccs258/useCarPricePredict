#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

def data_serch(data):
    data_search = DataSearch()
    df_data = data_search.load_data(data)
    data_search.status(df_data)




class DataSearch(object):

    def load_data(self,path):
        """ 加载数据"""
        df = pd.read_csv(path, sep=' ')
        print(df.describe())


        # 转换成数值类型
        df = df.apply(pd.to_numeric, errors='ignore')

        #返回列名以及缺失值统计
        columns = df.columns
        print(columns)
        map(lambda i:df[i].isna().sum())
        # for i in ['收入', '微博好友数', '消费理念']:
        #     num = df[i].isna().sum()
        #     print("特征%s的空值数量为：%d,占比为：%f" % (i, num, round(float(num) / total_num, 4)))


        return df


    def status(self,x):
        """
        统计数据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  最小值，最大值，均值，标准差，中位数，第一分位数，第三分位数，偏度（用pandas的skew()）
        """
        rst =  pd.Series([x.count(), x.min(), x.idxmin(), x.quantile(.25), x.median(),
                          x.quantile(.75), x.mean(), x.max(), x.idxmax(), x.mad(), x.var(),
                          x.std(), x.skew(), x.kurt()], index=['总数', '最小值', '最小值位置', '25%分位数',
                                                               '中位数', '75%分位数', '均值', '最大值', '最大值位数', '平均绝对偏差', '方差',
                                                               '标准差', '偏度', '峰度'])

        print("数据统计结果",rst)
        return rst



if __name__ == '__main__':
    train_data = r"C:\Users\ccs\Documents\dataWhale\used_car_train_20200313\used_car_train_20200313.csv"
    train_data_search =  data_serch(train_data)

