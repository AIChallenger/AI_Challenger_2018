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
Plant Disease Recognition is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python plant_disease_recognition2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is prediction_example.json, reference file is ref_example.json, test it by:
python plant_disease_recognition2018_eval.py --submit prediction_example.json --ref ref_example.json
The accuracy of the submited result, error message and warning message will be printed.
"""

import argparse
import json
import time


def evaluate(submit_file, reference_file):
    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        result_json = json.load(open(submit_file))

        f = open(reference_file)
        im_cls_list = json.load(f)
        f.close()

        if len(result_json) != len(im_cls_list):
            result['err_code'] = 1
            return result

        im_cls_dict = {}
        for each_im_cls in im_cls_list:
            im_name = each_im_cls.get('image_id')

            im_name = im_name.lower()
            if im_name[-4:] == '.jpg':
                im_name = im_name[0:-4]

            label_id = each_im_cls.get('disease_class')
            im_cls_dict[im_name] = label_id

        corrects = 0
        for each_item in result_json:
            image_id = each_item['image_id'].lower()
            if image_id[-4:] == '.jpg':
                image_id = image_id[0:-4]

            if int(each_item['disease_class']) == int(im_cls_dict[image_id]):
                corrects += 1
    except Exception as e:
        result['err_code'] = 1
        result['error'] = str(e)
        return result

    accuracy = corrects / len(im_cls_list)
    result['score'] = accuracy
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--submit',
        type=str,
        default='./prediction_example.json',
        help="""\
            Path to submitted file\
        """
    )

    parser.add_argument(
        '--ref',
        type=str,
        default='./ref_example.json',
        help="""
            Path to reference file
        """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = evaluate(args.submit, args.ref)
    print(time.time() - start_time)
    print(result)
