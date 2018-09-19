import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost
from sklearn.ensemble import RandomForestRegressor
import pickle as pk


def dump_data():
    data = nc.Dataset('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/data/BJRUC+AWS_100sta.nc')
    data = data.variables

    target = 't2m_obs'
    features = ['t2m_M', 'psfc_M', 'q2m_M', 'w10m_M', 'd10m_M', 'SWD_M', 'GLW_M', 'LH_M', 'HFX_M', 'RAIN_M',
                'PBLH_M']
    stations = list(range(100))

    train_days = list(range(658))
    test_days = list(range(658, 758, 1))

    train_fortimes = list(range(37))
    test_fortimes = list(range(37))
    all_train_data = {}
    all_test_data = {}

    for station in stations:
        train_X, train_y = extract_data(data=data,
                                        target=target,
                                        features=features,
                                        days=train_days,
                                        stations=[station],
                                        fortimes=train_fortimes)
        # 字典格式保存
        all_train_data[station] = (train_X, train_y)

        # 存储当前station的测试数据的字典
        test_station_data = {}
        for fortime in test_fortimes:
            temp_test_X, temp_test_y = extract_data(data=data,
                                                    target=target,
                                                    features=features,
                                                    days=test_days,
                                                    stations=[station],
                                                    fortimes=[fortime])

            test_station_data[fortime] = (temp_test_X, temp_test_y)
        # 把当前station的字典添加到总字典中
        all_test_data[station] = test_station_data
    # 保存
    pk.dump(all_train_data, open('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/station/train_data_YMF', 'wb'))
    pk.dump(all_test_data, open('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/station/test_data_YMF', 'wb'))


def load_data():
    train_data = pk.load(open('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/station/train_data_YMF', 'rb'))
    test_data = pk.load(open('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/station/test_data_YMF', 'rb'))
    return train_data, test_data


def check(data, key, threshold):
    """
    检查一个样本的某个特征是否合法
    :param data: 特征值的大小
    :param key: 对应的特征名称
    :param threshold: 字典,存放每个特征值对应的合理值区间
    :return: True or False
    """
    if not isinstance(data, np.float32):
        return False
    return threshold[key][0] <= data <= threshold[key][1]


def extract_data(data, target, features, days, stations, fortimes):
    threshold = {'psfc_M': [800, 1200],
                 'psur_obs': [800, 1200],

                 't2m_M': [-50, 50],
                 't2m_obs': [-50, 50],

                 'q2m_M': [0, 1500],
                 'q2m_obs': [0, 1500],

                 'w10m_M': [0, 30],
                 'w10m_obs': [0, 30],

                 'd10m_M': [0, 360],
                 'd10m_obs': [0, 360],

                 'RAIN_M': [0, 500],
                 'RAIN_obs': [0, 500],

                 'rh2m_obs': [0, 360],

                 'PBLH_M': [float('-inf'), float('+inf')],
                 'SWD_M': [float('-inf'), float('+inf')],
                 'GLW_M': [float('-inf'), float('+inf')],
                 'HFX_M': [float('-inf'), float('+inf')],
                 'LH_M': [float('-inf'), float('+inf')]}

    date = data['date'][:]
    date = [(int(str(i)[4:6]), int(str(i)[6:8])) for i in date]

    X, y = [], []

    # 每一个站点
    for stations_index in stations:
        print(stations_index, fortimes)
        station_X, station_y = [], []

        # 每一天
        for day in days:
            day_X, day_y = [], []

            # 每一个时刻
            for fortime in fortimes:
                cur_label = data[target][day, fortime, stations_index]
                if not check(cur_label, target, threshold=threshold):
                    continue

                # 为每一个样本加入三个新特征
                cur_features = [date[day][0], date[day][1], fortime]
                use_able = True

                for key in features:
                    single_feature = data[key][day, fortime, stations_index]
                    if check(single_feature, key, threshold=threshold):
                        cur_features.append(single_feature)
                    else:
                        # print(single_feature)
                        use_able = False
                        break

                # 如果该时刻可用就添加到当天的day_X和day_y中
                if use_able:
                    day_X.append(cur_features)
                    day_y.append(cur_label)

            # 如果当天的数据不为空
            if day_X and day_y:
                # print(day_X)
                # 转换为ndarray格式
                day_X = np.array(day_X).reshape((-1, len(features)+3))
                day_y = np.array(day_y).reshape((-1, 1))

                # 垂直拼接在本个站点的数据中
                station_X = day_X if station_X == [] else np.concatenate((station_X, day_X), axis=0)
                station_y = day_y if station_y == [] else np.concatenate((station_y, day_y), axis=0)

        X = station_X if X == [] else np.concatenate((X, station_X), axis=0)
        y = station_y if y == [] else np.concatenate((y, station_y), axis=0)

        # X = station_X if X == [] else np.concatenate((X, station_X), axis=0)
        # y = station_y if y == [] else np.concatenate((y, station_y), axis=0)

    return X, y


if __name__ == '__main__':

    data = nc.Dataset('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/data/BJRUC+AWS_100sta.nc')
    print(data.variables.keys())
    data = data.variables

    target = 't2m_obs'
    features = ['t2m_M', 'psfc_M', 'q2m_M', 'w10m_M', 'd10m_M', 'SWD_M', 'GLW_M', 'LH_M', 'HFX_M', 'RAIN_M', 'PBLH_M']
    stations = list(range(100))

    train_days = list(range(658))
    test_days = list(range(658, 758, 1))

    fortimes = list(range(37))

    # dump_data仅在训练数据需要发生变化时调用更新dump的数据
    # dump_data()

    train_data, test_data = load_data()


    # for item in [(0.1, 100, 5), (0.1, 100, 7), (0.1, 200, 5), (0.1, 200, 7), (0.1, 300, 5), (0.2, 100, 5), (0.2, 200, 7), (0.2, 300, 5), (0.2, 300, 7),]:
    #     learning_rate = item[0]
    #     n_estimators = item[1]
    #     max_depth = item[2]

    for learning_rate in [0.1, 0.2]:
        for n_estimators in [100, 200, 300]:
            for max_depth in [5, 7]:

                print(learning_rate, n_estimators, max_depth)
                for loop in range(1):
                    rmse_result = []
                    for station in stations:
                        station_rmse = []

                        temp_train_X, temp_train_y = train_data[station]

                        # # 参数设置
                        # learning_rate = 0.2
                        # n_estimators = 300
                        # max_depth = 5
                        subsample = 0.8

                        model = GradientBoostingRegressor(loss='huber',
                                                          learning_rate=learning_rate,
                                                          n_estimators=n_estimators,
                                                          criterion='friedman_mse',
                                                          max_depth=max_depth,
                                                          subsample=subsample,
                                                          )

                        model.fit(X=temp_train_X, y=temp_train_y)

                        # print(model.feature_importances_)

                        for fortime in fortimes:
                            temp_test_X, temp_test_y = test_data[station][fortime]

                            predict_y = model.predict(temp_test_X)

                            temp_rmse = np.sqrt(mean_squared_error(y_true=temp_test_y, y_pred=predict_y))

                            # print(station, fortime, temp_rmse)
                            station_rmse.append(temp_rmse)
                        rmse_result.append(station_rmse)

                    print(np.mean(rmse_result))
                    # pd.DataFrame(data=rmse_result).to_csv('/home/fengyuan/PycharmProjects/天气预测/use_forcast_info/station/GBDT_result' + str(loop) + '.csv')
