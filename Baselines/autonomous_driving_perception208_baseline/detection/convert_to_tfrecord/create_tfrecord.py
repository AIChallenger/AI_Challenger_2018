import argparse

from deepdrive_dataset.deepdrive_dataset_writer import DeepdriveDatasetWriter
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_FOLDS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_type', type=str, default='train', choices=DEEPDRIVE_FOLDS)
    parser.add_argument('--input_path', type=str, default='bdd100k',help='path to input dir of images and labels')
    parser.add_argument('--output_path', type=str, help='path to output tfrecord')
    FLAGS = parser.parse_args()

    dd = DeepdriveDatasetWriter()
    dd.write_tfrecord(FLAGS.fold_type, FLAGS.input_path, FLAGS.output_path)
