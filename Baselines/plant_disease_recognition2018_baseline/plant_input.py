import numpy as np
import cv2
import logging
import PIL.Image as Image
import json
import os

class plant_data_fn(object):

    def __init__(self, image_path, label_path):
        # get all the image name and its label
        self.data_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.data_dict[image['image_id']] = int(image['label_id'])
        self.start = 0
        self.end = 0
        self.Length = len(self.data_dict)
        self.img_name = list(self.data_dict.keys())
        self.image_path = image_path

    def img_resize(self, imgpath, img_size):
        # resize the image to the specific size
        img = Image.open(imgpath)
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
        # crop the proper size and scale to [-1, 1]
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        return img

    def next_batch(self, batch_size, img_size=32):
        # fetch a batch of images
        self.start = self.end
        if self.start >= self.Length:
            self.start = 0
        img_data = []
        img_label = []
        index = self.start
        while len(img_data) < batch_size:
            if index >= self.Length:
                index = 0
            img_data.append(self.img_resize(os.path.join(self.image_path, self.img_name[index]), img_size))
            img_label.append(self.data_dict[self.img_name[index]])
            index += 1   
        self.end = index
        return np.array(img_data), np.array(img_label)


def img_resize(imgpath, img_size):
        img = Image.open(imgpath)
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        return img


def train_log(filename='logfile'):
    # create logger
    logger_name = "filename"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    log_path = './' + filename + '.log'
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    # create formatter
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
return logger
