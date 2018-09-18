import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the trained model')
    parser.add_argument('image_dir', help='path to test image directory')
    parser.add_argument('output_dir', help='path to output directory of segmentation results')
    args = parser.parse_args()
    return args

def get_image_list(image_dir):
    files= os.listdir(image_dir)
    s = []
    for file in files:
        str_name = file[:21]
        s.append(str_name)
    return s


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        with open(model_path,'rb') as fd:
            graph_def = tf.GraphDef.FromString(fd.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            seg_map: Segmentation map of the input image.
        """
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(image)]
            })
        seg_map = batch_seg_map[0]
        return seg_map


def main():
    args = parse_args()

    model_path = args.model_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #load the model 
    model = DeepLabModel(model_path)
    image_list = get_image_list(image_dir)
    count = 0
    for i in range(len(image_list)):
        image_path = os.path.join(image_dir, image_list[i])
        orignal_im = Image.open(image_path)
        img_float = np.float32(orignal_im)
        #segmentation
        seg_map = model.run(img_float)
        count += 1
        if count % 1000 == 0:
            print('Finished', count)
        seg_image = seg_map.astype(np.uint8)
        #output the result
        res = cv2.resize(seg_image, dsize=orignal_im.size)
        cv2.imwrite(os.path.join(output_dir,image_list[i][:17]+'.png'), res)


if __name__ == '__main__':
    main()
