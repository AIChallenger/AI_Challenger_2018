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
Automated Segmentation of Retinal Edema Lesions is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python fundus_lesion2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is submit directory, reference file is ref directory, test it by:
python fundus_lesion2018_eval.py --submit ./submission_example.zip --ref ./groundtruth_example
The accuracy of the submited result, error message and warning message will be printed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import shutil
import argparse
import zipfile

import numpy as np
import traceback
from sklearn import metrics


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


def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
    """
    Classification task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 3)
    :param prediction: numpy matrix, (num_samples, 3)
    :param num_samples: int, default 128
    :return list:[AUC_1, AUC_2, AUC_3]
    """
    assert (ground_truth.shape == (num_samples, 3))
    assert (prediction.shape == (num_samples, 3))

    try:
        ret = [0.5, 0.5, 0.5]
        for i in range(3):
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
            ret[i] = metrics.auc(fpr, tpr)
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def test_aic_fundus_lesion_classification():
    import random
    ground_truth = np.asarray([random.randint(0, 1) for x in range(128 * 3)]).reshape((128, 3))
    prediction = np.asarray([random.random() for x in range(128 * 3)]).reshape((128, 3))

    assert (aic_fundus_lesion_classification(ground_truth, ground_truth) == [1.0, 1.0, 1.0])
    assert (aic_fundus_lesion_classification(ground_truth, -ground_truth) == [0.0, 0.0, 0.0])
    ret = aic_fundus_lesion_classification(ground_truth, prediction)
    assert (ret[0] < 0.6 and ret[1] < 0.6 and ret[2] < 0.6)


def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
    """
    Detection task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 1024, 512)
    :param prediction: numpy matrix, (num_samples, 1024, 512)
    :param num_samples: int, default 128
    :return list:[Dice_0, Dice_1, Dice_2, Dice_3]
    """
    assert (ground_truth.shape == (num_samples, 1024, 512))
    assert (prediction.shape == (num_samples, 1024, 512))

    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = 2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum())
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def test_aic_fundus_lesion_segmentation():
    import random
    ground_truth = np.asarray([random.randint(0, 3) for x in range(128 * 1024 * 512)]).reshape((128, 1024, 512))
    prediction = np.asarray([random.randint(0, 3) for x in range(128 * 1024 * 512)]).reshape((128, 1024, 512))

    assert (aic_fundus_lesion_segmentation(ground_truth, ground_truth) == [1.0, 1.0, 1.0, 1.0])
    assert (aic_fundus_lesion_segmentation(ground_truth, -ground_truth) == [0.0, 0.0, 0.0, 0.0])
    ret = aic_fundus_lesion_segmentation(ground_truth, prediction) == [0.0, 0.0, 0.0, 0.0]
    assert (ret[0] < 0.3 and ret[1] < 0.3 and ret[2] < 0.3 and ret[3] < 0.3)


def _eval_result(submit_path, ref_path):
    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }
    try:
        # cubes is 15 on real environment but 2 here for example
        cubes = 2

        # if submit file is zip file, upzip it
        submit_path = unzip_file(submit_path)

        ret_detection = []
        ret_segmentation = []
        submit_path_list_dir = os.listdir(submit_path)

        origin_submit_path = submit_path

        if len(submit_path_list_dir) == 1:
            submit_path = '{}/{}'.format(submit_path, os.listdir(submit_path)[0])
        cubes_list = os.listdir(ref_path)
        for cube in cubes_list:
            groud_truth = np.load(ref_path + '/%s' % cube)
            prediction_path = submit_path + '/%s' % cube
            if not os.path.exists(prediction_path):
                prediction_path = prediction_path.replace('_labelMark', '')
            prediction = np.load(prediction_path)
            if cube[-11:-4] == 'volumes':
                ret = aic_fundus_lesion_segmentation(groud_truth, prediction)
                ret_segmentation.append(ret)
            else:
                ret = aic_fundus_lesion_classification(groud_truth, prediction)
                ret_detection.append(ret)
        REA_detection, SRF_detection, PED_detection = 0.0, 0.0, 0.0
        REA_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
        n1, n2, n3, n4, n5, n6 = 0, 0, 0, 0, 0, 0
        for i in range(cubes):
            if not math.isnan(ret_detection[i][0]):
                REA_detection += ret_detection[i][0]
                n1 += 1
            if not math.isnan(ret_detection[i][1]):
                SRF_detection += ret_detection[i][1]
                n2 += 1
            if not math.isnan(ret_detection[i][2]):
                PED_detection += ret_detection[i][2]
                n3 += 1
            if not math.isnan(ret_segmentation[i][1]):
                REA_segementation += ret_segmentation[i][1]
                n4 += 1
            if not math.isnan(ret_segmentation[i][2]):
                SRF_segementation += ret_segmentation[i][2]
                n5 += 1
            if not math.isnan(ret_segmentation[i][3]):
                PED_segementation += ret_segmentation[i][3]
                n6 += 1

        # because of cubes on real environment is 15, here may have an error
        REA_detection /= n1
        SRF_detection /= n2
        PED_detection /= n3
        REA_segementation /= n4
        SRF_segementation /= n5
        PED_segementation /= n6
        avg_detection = (REA_detection + SRF_detection + PED_detection) / 3
        avg_segmentation = (REA_segementation + SRF_segementation + PED_segementation) / 3

        shutil.rmtree(origin_submit_path)
    except Exception as ex:
        shutil.rmtree(origin_submit_path)
        result['err_code'] = 1
        result['error'] = str(ex)
        return result
    result['score'] = (avg_detection + avg_segmentation) / 2
    result['score_extra'] = {
        'avg_detection': avg_detection,
        'avg_segementation': avg_segmentation
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--submit',
        type=str,
        default='./submission_example.zip',
        help="""\
                Path to submited file\
            """
    )

    parser.add_argument(
        '--ref',
        type=str,
        default='./groundtruth_example',
        help="""
                Path to reference file
            """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = _eval_result(args.submit, args.ref)
    print('Evaluation time of your result: %f s' % (time.time() - start_time))
    print(result)

