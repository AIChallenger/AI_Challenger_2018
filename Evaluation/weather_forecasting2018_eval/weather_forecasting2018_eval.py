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
python weather_forecasting2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH --anen ANEN_FILEPATH
A test case is provided, submited file is fore.csv, reference file is obs.csv, predicted by Institute of Urban Meteorology is anen.csv, test it by:
python weather_forecasting2018_eval.py --submit ./fore.csv --ref ./obs.csv --anen ./anen.csv
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


def _eval_result(fore_file, obs_file, anen_file):
    '''
    cal score
    :param fore_file: predicted by contestant
    :param obs_file: right answer
    :param anen_file: predicted by Institute of Urban Meteorology
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
        data_anen = pd.read_csv(anen_file, encoding='gbk')

        for i in data_obs:
            t = list(data_obs[data_obs[i] == -9999].index)
            data_obs = data_obs.drop(t)
            data_fore = data_fore.drop(t)
            data_anen = data_anen.drop(t)

        # 超算rmse
        t2m_rmse = rmse(data_obs['       t2m'], data_fore['       t2m'])
        rh2m_rmse = rmse(data_obs['      rh2m'], data_fore['      rh2m'])
        w10m_rmse = rmse(data_obs['      w10m'], data_fore['      w10m'])

        # anenrmse
        t2m_rmse1 = rmse(data_obs['       t2m'], data_anen['       t2m'])
        rh2m_rmse1 = rmse(data_obs['      rh2m'], data_anen['      rh2m'])
        w10m_rmse1 = rmse(data_obs['      w10m'], data_anen['      w10m'])

        # 降低率得分
        score_all = (score(t2m_rmse1, t2m_rmse) + score(rh2m_rmse1, rh2m_rmse) + score(w10m_rmse1, w10m_rmse)) / 3

        # bias得分
        score_bias_fore = score_bias(data_obs, data_fore)

        result['score'] = score_all
        result['score_extra'] = {
            'bias_fore': score_bias_fore
        }
    except Exception as e:
        result['err_code'] = 1
        result['warning'] = str(e)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--submit',
        type=str,
        default='./fore.csv',
        help="""\
                Path to submited file\
            """
    )

    parser.add_argument(
        '--ref',
        type=str,
        default='./obs.csv',
        help="""
                Path to reference file
            """
    )

    parser.add_argument(
        '--anen',
        type=str,
        default='./anen.csv',
        help="""
                    Path to anen file
                """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = _eval_result(args.submit, args.ref, args.anen)
    print(time.time() - start_time)
    print(result)
