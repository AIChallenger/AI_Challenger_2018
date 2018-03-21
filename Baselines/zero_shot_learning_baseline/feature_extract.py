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
This python script is to extract features for all images.
The command is:     python feature_extract.py Animals model/mobile_Animals_wgt.h5
The first parameter is the super-class.
The second parameter is the model weight.
The extracted features will be saved at 'features_Animals.pickle'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
import numpy as np
import pickle
import sys
import os


def main():
    # Parameters
    if len(sys.argv) == 3:
        superclass = sys.argv[1]
        model_weight = sys.argv[2]
    else:
        print('Parameters error')
        exit()

    # The constants
    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    date = '20180321'

    # Feature extraction model
    base_model = MobileNet(include_top=True, weights=None,
                           input_tensor=None, input_shape=None,
                           pooling=None, classes=classNum[superclass[0]])
    base_model.load_weights(model_weight)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('global_average_pooling2d_1').output)

    imgdir_train = '../zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_'+date\
                     +'/zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_images_'+date
    imgdir_test = '../zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_test_'+date
    categories = os.listdir(imgdir_train)
    categories.append('test')

    num = 0
    for eachclass in categories:
        if eachclass[0] == '.':
            continue
        if eachclass == 'test':
            classpath = imgdir_test
        else:
            classpath = imgdir_train+'/'+eachclass
        num += len(os.listdir(classpath))

    print('Total image number = '+str(num))

    features_all = np.ndarray((num, 1024))
    labels_all = list()
    images_all = list()
    idx = 0

    # Feature extraction
    for iter in tqdm(range(len(categories))):
        eachclass = categories[iter]
        if eachclass[0] == '.':
            continue
        if eachclass == 'test':
            classpath = imgdir_test
        else:
            classpath = imgdir_train+'/'+eachclass
        imgs = os.listdir(classpath)

        for eachimg in imgs:
            if eachimg[0] == '.':
                continue

            img_path = classpath+'/'+eachimg
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)

            features_all[idx, :] = feature
            labels_all.append(eachclass)
            images_all.append(eachimg)
            idx += 1

    features_all = features_all[:idx, :]
    labels_all = labels_all[:idx]
    images_all = images_all[:idx]
    data_all = {'features_all':features_all, 'labels_all':labels_all,
                'images_all':images_all}

    # Save features
    savename = 'features_' + superclass + '.pickle'
    fsave = open(savename, 'wb')
    pickle.dump(data_all, fsave)
    fsave.close()


if __name__ == "__main__":
    main()
