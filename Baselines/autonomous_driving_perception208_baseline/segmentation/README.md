# Drivable Area Segmentation Baseline for BDD100k Dataset
The baseline model for drivable area segmentation task is based on DeepLab v3+. The backbone of the deployed model is mobilenet_v2.

##Download Pretrained Model
First download the ckpt file of the pretrained deeplab v3+ model on Cityscapes dataset from the model zoo: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

##Convert Dataset into TFRecord Format
Use /datasets/build_voc2012_data.py to convert the input images into TFRecord format which is similar to the Pascal VOC dataset.

        python build_voc2012_data.py --image_folder path/to/image_dir --semantic_segmentation_folder path/to/drivable_area_ground_truth_image_dir --list_folder path/to/lists_for_training_images_names --output_dir path/to/save_converted_tfrecord_file


##Training
For training, use /train.py script

        python train.py --logtostderr --train_logdir path/to/save_log_files --tf_initial_checkpoint path/to/init_checkpoint --dataset_dir path/to/training_tfrecord_file

After training, use the script /export_model.py to export the model weights

        python export_model.py --checkpoint_path path/to/train_logdir --export_path path/to/save_exported_model_weights


##Inference
For inference 

1. cd inference

2. Use the script /inference.py to obtain the segmentation results

        python inference.py path/to/exported_model_dir path/to/image_dir path/to/output_dir


##Evaluation
For evaluation, use the official evaluation scripts which can be found at https://github.com/ucbdrive/bdd-data .

Specifically, use the script /bdd_data/evaluate.py to calculate the mIOU

        python evaluate.py drivable path/to/ground_truth_images path/to/predicted_images

