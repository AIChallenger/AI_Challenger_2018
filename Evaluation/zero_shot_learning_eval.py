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
Zero-shot learning is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python zero_shot_learning_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is submit.json, reference file is ref.json, test it by:
python zero_shot_learning_eval.py --submit ./pred.txt --ref ./ans.txt
The accuracy of the submited result, error message and warning message will be printed.
"""


import argparse
import time


def _load_data(submit_file, reference_file):
    # load submit result and reference result

    with open(submit_file, 'r') as f:
        submit_data = f.readlines()

    with open(reference_file, 'r') as f:
        ref_data = f.readlines()

    submit_dict = {}
    ref_dict = {}

    for each_line in submit_data:
        item = each_line.split()
        submit_dict[item[0]] = item[1]

    for each_line in ref_data:
        item = each_line.split()
        ref_dict[item[0]] = item[1]

    return submit_dict, ref_dict


def _eval_result(submit_file, reference_file):
    # eval the error rate

    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        submit_dict, ref_dict = _load_data(submit_file, reference_file)
    except Exception as e:
        result['err_code'] = 1
        result['error'] = str(e)
        return result

    right_count = 0

    keys = tuple(submit_dict.keys())
    for (key, value) in ref_dict.items():
        if key not in keys:
            result['warning'] = 'lacking image in your submitted result'
            print('warning: lacking image %s in your submitted result' % key)
            continue
        if submit_dict[key] == value:
            right_count += 1

    result['score'] = str(float(right_count) / len(ref_dict))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--submit',
        type=str,
        default='./pred.txt',
        help="""\
            Path to submited file\
        """
    )

    parser.add_argument(
        '--ref',
        type=str,
        default='./ans.txt',
        help="""
            Path to reference file
        """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = _eval_result(args.submit, args.ref)
    print('Evaluation time of your result: %f s' % (time.time() - start_time))
    print(result)
