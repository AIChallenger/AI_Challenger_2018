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
Opinion Questions Machine Reading Comprehension is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python oqmrc2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is text file, reference file is text file, test it by:
python oqmrc2018_eval.py --submit ./pred.txt --ref ./ans.txt
The accuracy of the submited result, error message and warning message will be printed.
"""


import argparse
import json
import time


def evaluate(prediction_file, dataset_file):
    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }
    try:
        predictions = {}
        with open(prediction_file, encoding='utf-8') as f1:
            for line in f1:
                qid, answer = line.strip("\n").split("\t", 2)
                predictions[int(qid)] = answer

        exact_match = 0
        total_data = 0
        with open(dataset_file, encoding='utf-8') as f2:
            for line in f2:
                total_data += 1
                data = json.loads(line.strip())
                qid = data["query_id"]
                answer = data["answer"]
                if qid in predictions:
                    if predictions[qid] == answer:
                        exact_match += 1
        result['score'] = float(exact_match)/total_data
        # print("accuracy=%s(%s/%s)" % (float(exact_match)/total_data, exact_match, total_data))
    except Exception as ex:
        result['err_code'] = 1
        result['error'] = str(ex)
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
    result = evaluate(args.submit, args.ref)
    print('Evaluation time of your result: %f s' % (time.time() - start_time))
    print(result)
