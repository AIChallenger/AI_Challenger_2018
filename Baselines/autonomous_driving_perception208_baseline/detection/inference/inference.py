import numpy as np
import os
import sys
sys.path.append('../')
import tensorflow as tf
import json
from PIL import Image
import cv2  
from utils import label_map_util
from utils import visualization_utils as vis_util
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the trained model')
    parser.add_argument('label_path', help='path to the class labels')
    parser.add_argument('image_dir', help='path to test image directory')
    parser.add_argument('output_dir', help='path to output detection results directory')
    args = parser.parse_args()

    return args

def get_image_list(image_dir):
    files= os.listdir(image_dir) 
    s = []
    for file in files: 
        str_name = file[:21]
        s.append(str_name) 
    return s

def load_model(model_path,label_path):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph,category_index

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def write_results_into_single_list(image, boxes, classes, scores, category_index, name, use_normalized_coordinates=False):
  """Output the detection result into a single json file to be evaluated with the ground truth
  Args:
    image: float numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None. 
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    name: the name of the input images
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
  """
  out = list()
  for i in range(boxes.shape[0]):
    box = tuple(boxes[i].tolist())
    ymin, xmin, ymax, xmax = box
    (im_width, im_height) = image.size
    if use_normalized_coordinates:
        (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    class_name = category_index[classes[i]]['name']
    score_f = float(scores[i])
    if score_f < 0.5:
        continue
    xmin_f = float(xmin) 
    xmax_f = float(xmax) 
    ymin_f = float(ymin)
    ymax_f = float(ymax)
    tmp = {'name': name,'timestamp': 10000,'category': class_name,'bbox': [xmin_f, ymin_f, xmax_f, ymax_f],'score': score_f}
    out.append(tmp)
  return out

def main():
    args = parse_args()

    model_path = args.model_path
    label_path = args.label_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    result = []
    count = 0
    detection_graph,category_index = load_model(model_path,label_path)
    # # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_list = get_image_list(image_dir)
            for i in range(len(image_list)):
                image_path = os.path.join(image_dir,image_list[i]) 
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                #output the detection result into a single json file
                out = write_results_into_single_list(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, image_list[i][:17], use_normalized_coordinates=True)
                result.extend(out) 
                count += 1
                if count % 1000 == 0:
                    print('Finished', count)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
                cv2.imwrite(os.path.join(output_dir,image_list[i]), image_np)

        with open('./prediction.json', 'w') as fp:
            json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
