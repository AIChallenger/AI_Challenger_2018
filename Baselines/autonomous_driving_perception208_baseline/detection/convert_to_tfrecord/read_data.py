import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle

from deepdrive_dataset.deepdrive_dataset_reader import DeepdriveDatasetReader
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_FOLDS, DEEPDRIVE_VERSIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch Size to use', default=4)
    parser.add_argument('--fold_type', type=str, choices=DEEPDRIVE_FOLDS, default='train')
    parser.add_argument('--version', type=str, default='100k', choices=DEEPDRIVE_VERSIONS)
    FLAGS = parser.parse_args()

    reader = DeepdriveDatasetReader(batch_size=FLAGS.batch_size)
    if FLAGS.fold_type == 'train':
        iterator = reader.load_train_data_bbox(FLAGS.version, False)
    elif FLAGS.fold_type == 'val':
        iterator = reader.load_val_data_bbox(FLAGS.version, False)
    elif FLAGS.fold_type == 'test':
        iterator = reader.load_test_data_bbox(FLAGS.version, False)
    else:
        raise BaseException('Unknown fold type: {0}'.format(FLAGS.fold_type))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(iterator)
        out_labels = DeepdriveDatasetReader.parsing_boundingboxes(None, 'labels')
        out_dict = dict(zip(out_labels, out))

        # Visualize the output
        import math
        images = out_dict['image']
        fig_row = int(math.ceil(images.shape[0] / 2))
        fig_col = int(math.ceil(images.shape[0] / 2))
        f, axarr = plt.subplots(nrows=fig_row, ncols=fig_col)
        for im_id in range(0, images.shape[0]):
            plt.sca(axarr[im_id % 2, int(im_id/2)])
            plt.imshow(images[im_id, :])
            # Boundingbox order is ymin, xmin, ymax, xmax
            for box in out_dict['bboxes'][im_id, :]:
                rect = Rectangle(
                    xy=(box[1], box[0]), width=box[3] - box[1], height=box[2] - box[0], fill=False)
                axarr[im_id % 2, int(im_id/2)].add_patch(rect)

            #
            axarr[im_id % 2, int(im_id/2)].set_xticks([])
            axarr[im_id % 2, int(im_id/2)].set_yticks([])
            axarr[im_id % 2, int(im_id/2)].set_ylabel('Batch-Id: {0}'.format(im_id))
            axarr[im_id % 2, int(im_id/2)].set_xlabel(str(out_dict['image_ids'][im_id]))
        plt.show()
