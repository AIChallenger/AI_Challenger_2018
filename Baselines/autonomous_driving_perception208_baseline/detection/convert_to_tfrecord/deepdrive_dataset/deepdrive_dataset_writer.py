import os
import re
import json
from os.path import expanduser
import hashlib
import zipfile
import datetime
import tensorflow as tf
import numpy as np

from .utils import mkdir_p
from .deepdrive_dataset_download import DeepdriveDatasetDownload
from .deepdrive_versions import DEEPDRIVE_LABELS
from .tf_features import *
from PIL import Image


class DeepdriveDatasetWriter(object):
    feature_dict = {
        'image/height': None,
        'image/width': None,
        'image/object/bbox/id': None,
        'image/object/bbox/xmin': None,
        'image/object/bbox/xmax': None,
        'image/object/bbox/ymin': None,
        'image/object/bbox/ymax': None,
        'image/object/bbox/truncated': None,
        'image/object/bbox/occluded': None,
        'image/object/class/label/name': None,
        'image/object/class/label/id': None,
        'image/object/class/label': None,
        'image/encoded': None,
        'image/format': None,
        'image/id': None,
        'image/source_id': None,
        'image/filename': None,
        'image/key/sha256':None,
        'image/object/is_crowd':None,
        'image/object/area':None,   
}


    @staticmethod
    def feature_dict_description(type='feature_dict'):
        """
        Get the feature dict. In the default case it is filled with all the keys and the items set to None. If the
        type=reading_shape the shape description required for reading elements from a tfrecord is returned)
        :param type: (anything = returns the feature_dict with empty elements, reading_shape = element description for
        reading the tfrecord files is returned)
        :return:
        """
        obj = DeepdriveDatasetWriter.feature_dict
        if type == 'reading_shape':
            obj['image/height'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/width'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/object/bbox/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/bbox/xmin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/xmax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/truncated'] = tf.VarLenFeature(tf.string)
            obj['image/object/bbox/occluded'] = tf.VarLenFeature(tf.string)
            obj['image/encoded'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/format'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/filename'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/source_id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/object/class/label/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label/name'] = tf.VarLenFeature(tf.string)
            obj['image/key/sha256'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/object/is_crowd'] = tf.FixedLenFeature((), tf.int64, 0)
            obj['image/object/area'] = tf.VarLenFeature(tf.float32)
        return obj

    def __init__(self):
        pass

    def get_image_label_folder(self, fold_type=None, input_path=None):
        """
        Returns the folder containing all images and the folder containing all label information
        :param fold_type:
        :param input_path:
        :return: Raises BaseExceptions if expectations are not fulfilled
        """
        assert(fold_type in ['train', 'test', 'val'])

        labels_path = os.path.join(input_path,'labels')
        full_images_path = os.path.join(input_path, 'images','100k', fold_type)

        if fold_type == 'test':
            return full_images_path, None
        else:
            label_name = 'bdd100k_labels_images_'+str(fold_type)+'.json'
            full_labels_path = os.path.join(labels_path, label_name)
            return full_images_path, full_labels_path

    def _get_boundingboxes(self, annotations_for_picture_id):
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded,area =\
            [], [], [], [], [], [], [], [], [], []
        if annotations_for_picture_id is None:
            return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded,area
        for obj in annotations_for_picture_id['labels']:
            if 'box2d' not in obj:
                    continue
            boxid.append(obj['id'])
            xmin.append(obj['box2d']['x1'])
            xmax.append(obj['box2d']['x2'])
            ymin.append(obj['box2d']['y1'])
            ymax.append(obj['box2d']['y2'])
            area_tmp = abs(obj['box2d']['x1']-obj['box2d']['x2'])*abs(obj['box2d']['y1']-obj['box2d']['y2'])
            area.append(area_tmp)
            label.append(obj['category'])
            label_id.append(DEEPDRIVE_LABELS.index(obj['category']) + 1)
            attributes = obj['attributes']
            truncated.append(attributes.get('truncated', False))
            occluded.append(attributes.get('occluded', False))
        return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded, area


    def _get_tf_feature_dict(self, image_path, annotations):
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded,area = \
            self._get_boundingboxes(annotations)
        truncated = np.asarray(truncated)
        occluded = np.asarray(occluded)

        # convert things to bytes
        label_bytes = [tf.compat.as_bytes(l) for l in label]
        image_file = os.path.join(image_path,annotations["name"])
        im = Image.open(image_file)
        image_width, image_height = im.size
        norm_xmin = [float(x)/image_width for x in xmin]
        norm_xmax = [float(x)/image_width for x in xmax]
        norm_ymin = [float(y)/image_height for y in ymin]
        norm_ymax = [float(y)/image_height for y in ymax]
        norm_area = [float(a)/(image_width*image_height) for a in area]
        image_filename = annotations["name"]
        image_fileid = re.search('^(.*)(\.jpg)$', image_filename).group(1)
        image_format = 'jpg'
        tmp_feat_dict = DeepdriveDatasetWriter.feature_dict
        tmp_feat_dict['image/id'] = bytes_feature(image_fileid.encode())
        tmp_feat_dict['image/source_id'] = bytes_feature(image_fileid.encode())
        tmp_feat_dict['image/height'] = int64_feature(image_height)
        tmp_feat_dict['image/width'] = int64_feature(image_width)
        with tf.gfile.GFile(image_file, 'rb') as fid:
            encoded_jpg = fid.read()
        key = hashlib.sha256(encoded_jpg).hexdigest()
        tmp_feat_dict['image/encoded'] = bytes_feature(encoded_jpg)
        tmp_feat_dict['image/key/sha256'] = bytes_feature(key.encode())
        tmp_feat_dict['image/object/is_crowd'] = int64_feature(0)   
        tmp_feat_dict['image/format'] = bytes_feature(image_format.encode())
        tmp_feat_dict['image/filename'] = bytes_feature(image_filename.encode())
        tmp_feat_dict['image/object/bbox/id'] = int64_feature(boxid)
        tmp_feat_dict['image/object/bbox/xmin'] = float_feature(norm_xmin)
        tmp_feat_dict['image/object/bbox/xmax'] = float_feature(norm_xmax)
        tmp_feat_dict['image/object/bbox/ymin'] = float_feature(norm_ymin)
        tmp_feat_dict['image/object/bbox/ymax'] = float_feature(norm_ymax)
        tmp_feat_dict['image/object/bbox/truncated'] = bytes_feature(truncated.tobytes())
        tmp_feat_dict['image/object/bbox/occluded'] = bytes_feature(occluded.tobytes())
        tmp_feat_dict['image/object/class/label/id'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label/name'] = bytes_feature(label_bytes)
        tmp_feat_dict['image/object/area'] = float_feature(norm_area)
        return tmp_feat_dict


    def _get_tf_feature(self, image_path, annotations):
        feature_dict = self._get_tf_feature_dict(image_path, annotations)
        return tf.train.Features(feature=feature_dict)

    def write_tfrecord(self, fold_type=None, input_path=None, output_path=None):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        full_images_path, full_labels_path = self.get_image_label_folder(fold_type, input_path)
        count = 1
        # get the files
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_filename = fold_type + '.tfrecord'
            with tf.python_io.TFRecordWriter(os.path.join(output_path,output_filename)) as writer:
                with open(full_labels_path,'r') as f:
                    groundtruth_data = json.loads(f.read())
                    for annotations in groundtruth_data:
                        try:
                            feature = self._get_tf_feature(full_images_path, annotations)
                            example = tf.train.Example(features=feature)
                            writer.write(example.SerializeToString())
                            print('converting:',annotations["name"],'total_nums:',count)
                        except:
                            print('when converting:',annotations["name"],'meets error')
                            count -= 1
                        finally:
                            count += 1
