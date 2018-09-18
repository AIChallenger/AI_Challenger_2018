# Object Detection Baseline for BDD100k Dataset
The baseline model for object detection task is based on faster-rcnn model. The feature extractor of the baseline model is ResNet-50.


## Install Tensorflow Object Detection API
First install Tensorflow Object Detection API. The detailed steps can be found at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md


## Download the pretrained model
Download the ckpt file of the pretrained model on coco dataset from the model zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md


## Convert the dataset into TFrecord 
We refer to https://github.com/meyerjo/deepdrive_dataset_tfrecord to convert our training images into TFRecord format and slightly change the scripts to fit our model.

1. Download the zipfile of images and labes

2. Unzip both zip files into same folder

3. Use the script /convert_to_tfrecord/create_tfrecord.py in order to create the TFRecord file

        python create_tfrecord.py --fold_type train --input_path path/to/input_dir --output_path path/to/output_tfrecord_path


## Train the model
For training, run the training script

        python train.py --logtostderr  --train_dir path/to/train_dir --model_config_path path/to/model_config.pbtxt --train_config_path path/to/train_config.pbtxt --input_config_path path/to/train_input_config.pbtxt

After training, use the following script to export the inference graph

        python export_inference_graph --input_type image_tensor  --pipeline_config_path path/to/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix path/to/model.ckpt --output_directory path/to/exported_model_directory


## Inference 
For inference,

1. cd inference

2. Use the script

        python inference.py path/to/model path/to/class_labels path/to/input_images_dir path/to/output_dir

3. The visualization of the detection results can be found in path/to/output_directory and the json file for evaluation is in the current directory named prediction.json.


## Evaluation
For evaluation, we use the official evaluation scripts which can be found at https://github.com/ucbdrive/bdd-data . 
 
1. Use /bdd_data/label2det.py to prepare the ground truth labels from seperate json files into single json file

        python label2det.py path/to/separate_label_directory /single_json_file_name

2. Use the script /bdd_data/evaluate.py to calculate the mAP

        python evaluate.py det path/to/ground_truth_label_json_file path/to/predicted_label_json_file

