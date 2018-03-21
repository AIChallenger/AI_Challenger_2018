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
Baseline codes for zero-shot learning task.
This python script is the baseline to implement zero-shot learning on each super-class.
The command is:     python MDP.py Animals
The only parameter is the super-class name.
This method is from the paper
@inproceedings{zhao2017zero,
  title={Zero-shot learning posed as a missing data problem},
  author={Zhao, Bo and Wu, Botong and Wu, Tianfu and Wang, Yizhou},
  booktitle={Proceedings of ICCV Workshop},
  pages={2616--2622},
  year={2017}
}
Cite the paper, if you use this code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
import numpy as np
import pickle
import sys

# Convert strings to attributes
def attrstr2list(s):
    s = s[1:-2]
    tokens = s.split()
    attrlist = list()
    for each in tokens:
        attrlist.append(float(each))
    return attrlist


def main():
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Parameters error')
        exit()

    file_feature = 'features_'+superclass+'.pickle'

    # The constants
    if superclass[0] == 'H':
        classNum = 30
    else:
        classNum = 50
    testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    date = '20180321'

    # Load seen/unseen split
    label_list_path = '../zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_'+date\
            + '/zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_annotations_'+'label_list_'+date+'.txt'
    fsplit = open(label_list_path, 'r', encoding='UTF-8')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_train = list()
    names_train = list()
    for each in lines_label:
        tokens = each.split(', ')
        list_train.append(tokens[0])
        names_train.append(tokens[1])
    list_test = list()
    for i in range(classNum):
        label = 'Label_' + superclass[0] + '_' + str(i+1).zfill(2)
        if label not in list_train:
            list_test.append(label)

    # Load attributes
    attrnum = {'A':123, 'F':58, 'V':81, 'E':75, 'H':22}

    attributes_per_class_path = '../zsl_' + testName[superclass[0]] +'_' + str(superclass).lower() + '_train_' + date \
            + '/zsl_' + testName[superclass[0]] +'_' + str(superclass).lower() \
            + '_train_annotations_' + 'attributes_per_class_' + date + '.txt'
    fattr = open(attributes_per_class_path, 'r', encoding='UTF-8')
    lines_attr = fattr.readlines()
    fattr.close()
    attributes = dict()
    for each in lines_attr:
        tokens = each.split(', ')
        label = tokens[0]
        attr = attrstr2list(tokens[1])
        if not (len(attr) == attrnum[superclass[0]]):
            print('attributes number error\n')
            exit()
        attributes[label] = attr

    # Load image features
    fdata = open(file_feature, 'rb')
    features_dict = pickle.load(fdata)  # variables come out in the order you put them in
    fdata.close()
    features_all = features_dict['features_all']
    labels_all = features_dict['labels_all']
    images_all = features_dict['images_all']

    # Label mapping
    for i in range(len(labels_all)):
        if labels_all[i][2:] in names_train:
            idx = names_train.index(labels_all[i][2:])
            labels_all[i] = list_train[idx]

    # Calculate prototypes (cluster centers)
    features_all = features_all/np.max(abs(features_all))
    dim_f = features_all.shape[1]
    prototypes_train = np.ndarray((int(classNum/5*4), dim_f))

    dim_a = attrnum[superclass[0]]
    attributes_train = np.ndarray((int(classNum/5*4), dim_a))
    attributes_test = np.ndarray((int(classNum/5*1), dim_a))

    for i in range(len(list_train)):
        label = list_train[i]
        idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
        prototypes_train[i, :] = np.mean(features_all[idx, :], axis=0)
        attributes_train[i, :] = np.asarray(attributes[label])

    for i in range(len(list_test)):
        label = list_test[i]
        attributes_test[i, :] = np.asarray(attributes[label])

    # Structure learning
    LASSO = models.Lasso(alpha=0.01)
    LASSO.fit(attributes_train.transpose(), attributes_test.transpose())
    W = LASSO.coef_

    # Image prototype synthesis
    prototypes_test = (np.dot(prototypes_train.transpose(), W.transpose())).transpose()

    # Prediction
    label = 'test'
    idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
    features_test = features_all[idx, :]
    images_test = [images_all[i] for i in idx]
    prediction = list()

    for i in range(len(idx)):
        temp = np.repeat(np.reshape((features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
        distance = np.sum((temp - prototypes_test)**2, axis=1)
        pos = np.argmin(distance)
        prediction.append(list_test[pos])

    # Write prediction
    fpred = open('pred_'+ superclass + '.txt', 'w')

    for i in range(len(images_test)):
        fpred.write(str(images_test[i])+' '+prediction[i]+'\n')
    fpred.close()


if __name__ == "__main__":
    main()
