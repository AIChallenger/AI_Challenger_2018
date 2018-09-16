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
Autonomous Driving Perception is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python autonomous_driving_perception2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is submit directory, reference file is ref directory, test it by:
python autonomous_driving_perception2018_eval.py --submit ./submission.zip --ref ./ref
The accuracy of the submited result, error message and warning message will be printed.
"""

import argparse
import requests
import json
import os
import shutil
import time
import zipfile
import os.path as osp

from . import evaluate


def unzip_file(file_path):
    '''
    Unzip file
    :param file_path:
    :return:
    '''
    result_file_path = file_path
    if os.path.isdir(file_path):
        pass
    else:
        if '/' in file_path:
            filename = file_path.split('/')[-1]
        else:
            filename = file_path
        suffix = filename.split('.')[-1]
        if suffix == 'zip':
            zip_ref = zipfile.ZipFile(file_path, 'r')
            result_file_path = '{filename}_files'.format(**locals())
            if os.path.isdir(result_file_path):
                pass
            else:
                os.mkdir(result_file_path)
            for names in zip_ref.namelist():
                zip_ref.extract(names, result_file_path)
            zip_ref.close()
    return result_file_path


def get_detection_scores(gt_path, prediction_path):
    try:
        mAP, _ = evaluate.evaluate_detection(gt_path, prediction_path)
        status = 2
    except:
        status = 0
        mAP = 0
    finally:
        return mAP, status


def get_segmentation_scores(gt_path, prediction_path):
    try:
        mIOU, _ = evaluate.evaluate_drivable(gt_path, prediction_path)
        status = 2
    except:
        status = 0
        mIOU = 0
    finally:
        return mIOU, status


def eval_result(submit_path, ref_path):
    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        # if submit file is zip file, upzip it
        submit_path = unzip_file(submit_path)
        origin_submit_path = submit_path
        submit_path_dir_list = os.listdir(submit_path)
        if '__MACOSX' in submit_path_dir_list:
            submit_path_dir_list.remove('__MACOSX')
        if 'det.json' not in submit_path_dir_list:
            if len(submit_path_dir_list) == 1:
                submit_path = osp.join(submit_path, submit_path_dir_list[0])
            else:
                raise Exception('Wrong!')

        det_gt_path = osp.join(ref_path, 'det.json')
        seg_gt_path = osp.join(ref_path, 'seg')
        det_prediction_path = osp.join(submit_path, 'det.json')
        seg_prediction_path = osp.join(submit_path, 'seg')

        mAP, status_det = get_detection_scores(det_gt_path, det_prediction_path)
        # get mIOU for the drivable area segmentation task
        mIOU, status_seg = get_segmentation_scores(seg_gt_path, seg_prediction_path)

        if status_det == 2 or status_seg == 2:
            # everything works fine
            status = 2
        else:
            # something wrong with user's submission.
            status = 0
        result['score'] = 0
        result['score_extra'] = {
            'mAP': round(mAP, 6),
            'mIoU': round(mIOU, 6)
        }
        shutil.rmtree(origin_submit_path)
    except Exception as ex:
        result['err_code'] = 1
        result['error'] = str(ex)
    return result


def main():
    parser = argparse.ArgumentParser()

    # parse the input args
    parser.add_argument(
        '--submit',
        type=str,
        default='./submission.zip',
        help="""
            Path to submited file
        """
    )

    parser.add_argument(
        '--ref',
        type=str,
        default='./ref',
        help="""
            Path to reference file
        """
    )
    args = parser.parse_args()
    start_time = time.time()
    result = eval_result(args.submit_path, args.ref)
    print('Evaluation time of your result: %f s' % (time.time() - start_time))
    print(result)


if __name__ == '__main__':
    main()
