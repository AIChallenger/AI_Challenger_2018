#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Weather Forecasting is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python weather_forecasting2018_eval.py --submit SUBMIT_FILEPATH --obs OBSERVATION_FILEPATH --fore RMAPS_FILEPATH
A test case is provided, submited file is anene.csv, observation file is obs.csv, RMAPS result is anen.csv, test it by:
python weather_forecasting2018_eval.py --submit ./anen.csv --obs ./obs.csv --fore ./fore.csv
The accuracy of the submited result, error message and warning message will be printed.
"""


import time
import argparse
from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error


def bias(a, b):
    error = 0
    for i in a.index:
        error = error + (a[i] - b[i])
    err = error / len(a)
    return err


def rmse(a, b):
    return sqrt(mean_squared_error(a, b))


def score(a, b):
    return (b - a) / b


def score_bias(data_obs, data_fore):
    y_name = ['       t2m', '      rh2m', '      w10m']
    score = 0
    for i in y_name:
        score = score + bias(data_obs[i], data_fore[i])

    score = score / 3
    return score


def delete_non_value(data_obs, data_fore, data_anen, column):
    t = list(data_obs[data_obs[column] == -9999].index)
    data_obs_t2m = data_obs[column].drop(t)
    data_fore_t2m = data_fore[column].drop(t)
    data_anen_t2m = data_anen[column.strip()].drop(t)
    return data_obs_t2m, data_fore_t2m, data_anen_t2m


def _eval_result(fore_file, obs_file, anen_file):
    '''
    cal score
    :param fore_file: 超算结果
    :param obs_file: 正确答案
    :param anen_file: 选手提交结果
    :return:
    '''
    # eval the error rate

    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        data_obs = pd.read_csv(obs_file, encoding='gbk')
        data_fore = pd.read_csv(fore_file, encoding='gbk')

        delimiter_list = [',', ';', ' ,', '    ', '  ', ' ', '\t']
        data_anen_columns_list = []
        for each_delimiter in delimiter_list:
            data_anen = pd.read_csv(anen_file, encoding='gbk', delimiter=each_delimiter)
            old_data_anen_columns_list = list(data_anen.columns)
            data_anen_columns_list = [_ for _ in old_data_anen_columns_list if _.strip() != '']
            if len(data_anen_columns_list) == 4:
                break

        no_list = [_.strip() for _ in data_obs['  OBS_data']]
        for each_no in no_list:
            # if each_no.split('_')[1] in set():
            if each_no.split('_')[1] in set(['00', '01', '02', '03']):
                real_no = '  {each_no}'.format(**locals())
                this_index = list(data_obs[data_obs['  OBS_data'] == real_no].index)
                data_obs = data_obs.drop(this_index)
                data_fore = data_fore.drop(this_index)
                data_anen = data_anen.drop(this_index)

        # 超算rmse
        data_anen_dict = {}
        for each_anen_column in data_anen_columns_list:
            data_anen_dict[each_anen_column.strip()] = data_anen[each_anen_column]

        data_obs_t2m, data_fore_t2m, data_anen_t2m = delete_non_value(data_obs, data_fore, data_anen_dict, '       t2m')
        data_obs_rh2m, data_fore_rh2m, data_anen_rh2m = delete_non_value(data_obs, data_fore, data_anen_dict, '      rh2m')
        data_obs_w10m, data_fore_w10m, data_anen_w10m = delete_non_value(data_obs, data_fore, data_anen_dict, '      w10m')

        # t2m_rmse = rmse(data_obs['       t2m'], data_fore['       t2m'])
        # rh2m_rmse = rmse(data_obs['      rh2m'], data_fore['      rh2m'])
        # w10m_rmse = rmse(data_obs['      w10m'], data_fore['      w10m'])
        #
        # # anenrmse
        # t2m_rmse1 = rmse(data_obs['       t2m'], data_anen_dict['t2m'])
        # rh2m_rmse1 = rmse(data_obs['      rh2m'], data_anen_dict['rh2m'])
        # w10m_rmse1 = rmse(data_obs['      w10m'], data_anen_dict['w10m'])
        t2m_rmse = rmse(data_obs_t2m, data_fore_t2m)
        rh2m_rmse = rmse(data_obs_rh2m, data_fore_rh2m)
        w10m_rmse = rmse(data_obs_w10m, data_fore_w10m)

        # anenrmse
        t2m_rmse1 = rmse(data_obs_t2m, data_anen_t2m)
        rh2m_rmse1 = rmse(data_obs_rh2m, data_anen_rh2m)
        w10m_rmse1 = rmse(data_obs_w10m, data_anen_w10m)

        # # anenrmse
        # t2m_rmse1 = rmse(data_obs['       t2m'], data_anen['       t2m'])
        # rh2m_rmse1 = rmse(data_obs['      rh2m'], data_anen['      rh2m'])
        # w10m_rmse1 = rmse(data_obs['      w10m'], data_anen['      w10m'])

        # 降低率得分
        score_all = (score(t2m_rmse1, t2m_rmse) + score(rh2m_rmse1, rh2m_rmse) + score(w10m_rmse1, w10m_rmse)) / 3

        # bias得分
        score_bias_fore = score_bias(data_obs, data_fore)

        result['score'] = score_all
        result['score_extra'] = {
            'bias_fore': score_bias_fore,
            't2m_rmse': t2m_rmse1,
            'rh2m_rmse': rh2m_rmse1,
            'w10m_rmse': w10m_rmse1
        }

        print('t2m', 'rh2m', 'w10m')
        print(t2m_rmse, rh2m_rmse, w10m_rmse)
        print(t2m_rmse1, rh2m_rmse1, w10m_rmse1)
        print(score_all)
        print(score_bias_fore)
    except Exception as e:
        result['err_code'] = 1
        result['warning'] = str(e)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--submit',
        type=str,
        default='./anen.csv',
        help="""\
                Path to submited file\
            """
    )

    parser.add_argument(
        '--obs',
        type=str,
        default='./obs.csv',
        help="""
            Path to true result file
        """
    )

    parser.add_argument(
        '--fore',
        type=str,
        default='./fore.csv',
        help="""
            Path to RMAPS file
        """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = _eval_result(args.fore, args.obs, args.submit)
    print(time.time() - start_time)
    print(result)
