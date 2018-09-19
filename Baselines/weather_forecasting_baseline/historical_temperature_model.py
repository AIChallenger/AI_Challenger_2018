# -*- coding:UTF-8 -*-
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import time
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
#import xgboost
#from sklearn.ensemble import RandomForestRegressor
import pickle as pk

nc_data = nc.Dataset('../../data/BJRUC+AWS_100sta.nc')
nc_data = nc_data.variables

# all_data_csv_keys = ['station_id', 'station_value', 'day_id', 'day_value', 'foretimes', 'psfc_M', 't2m_M', 'q2m_M', 'w10m_M',
#               'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'psur_obs',
#               't2m_obs', 'q2m_obs', 'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

all_data = pd.read_csv('../../data/all_data.csv')

#result = []
for station_id in range(0, 100):
    print(station_id,'+++++++++++++++++++++++++++++++++++')
    all_data.is_copy = False
    data = all_data[all_data['station_id'] == station_id]
    data.reset_index(inplace=True, drop=True)
    data['history_t2m_obs'] = -9999
    data['month'] = [str(item)[4:6] for item in data['day_value']]
    data['day'] = [str(item)[6:8] for item in data['day_value']]

    # make new features
    for i in range(data.shape[0]):
        if i%3000==0:
            print(i)
        day_id = data.loc[i, 'day_id']
        if day_id-2 < 0:
            continue

        foretime = data.loc[i, 'foretimes']

        data.loc[i, 'history_t2m_obs'] = nc_data['t2m_obs'][day_id-2, foretime, station_id]

    data = data.replace('--', -9999)
    data = data.astype(float)

    # save midile result
    data.to_csv('./station_' + str(station_id) + '_all.csv', index=False)

    """
    #data = pd.read_csv('./station_0_history.csv')

    #
    avaliable_data = data[(data['psfc_M'] <= 1200) & (data['psfc_M'] >= 800)
                          & (data['t2m_M'] <= 50) & (data['t2m_M'] >= -50)
                          & (data['q2m_M'] <= 1500) & (data['q2m_M'] >= 0)
                          & (data['w10m_M'] <= 30) & (data['w10m_M'] >= 0)
                          & (data['d10m_M'] <= 360) & (data['d10m_M'] >= 0)
                          & (data['RAIN_M'] <= 500) & (data['RAIN_M'] >= 0)
                          & (data['t2m_obs'] <= 50) & (data['t2m_obs'] >= -50)
                          & (data['history_t2m_obs'] <= 50) & (data['history_t2m_obs'] >= -50)]

    #
    print('划分训练集和测试集')
    train_data = avaliable_data[(avaliable_data['day_id'] >= 0) & (avaliable_data['day_id'] <= 657)]
    test_data = avaliable_data[avaliable_data['day_id'] > 657]

    feature = ['sta#tion_id', 'month', 'day', 'foretimes', 'psfc_M', 't2m_M', 'q2m_M', 'w10m_M',
                  #'d10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'history_t2m_obs']
    target = ['t2m_obs']

    #
    learning_rate = 0.1
    n_estimators = 100
    max_depth = 5
    subsample = 0.8

    model = GradientBoostingRegressor(loss='huber',
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      criterion='friedman_mse',
                                      max_depth=max_depth,
                                      subsample=subsample,
                                      )
    # 残差 = 真实值-超算值
    error = train_data[target].reshape((-1, )) - train_data['t2m_M']

    print('开始训练模型')
    model.fit(X=train_data[feature], y=error)

    station_rmse = []

    for foretime in range(37):
        temp_test_data = test_data[test_data['foretimes'] == foretime]

        temp_test_feature = temp_test_data[feature]
        temp_test_target = temp_test_data[target]

        predict_error = model.predict(X=temp_test_feature)

        # 拟合的残差值+超算值才是最终预测温度
        temp_rmse = np.sqrt(mean_squared_error(y_true=temp_test_target, y_pred=error+temp_test_feature['t2m_M']))
        station_rmse.append(temp_rmse)

    result.append(station_rmse)

print(np.mean(result))
pd.DataFrame(result).to_csv('./history_error_result.csv')
"""
