# AI_Challenger

Neural Machine Translation (English-to-Chinese) baseline for AI_Challenger dataset.

# Requirenments

- python 2.7
- TensorFlow 1.2.0
- tensor2tensor
- jieba 0.39

For reference, we used the 1.8.0 version of tensor2tensor, the 9.0.176 version of cuda, and the 7.0.5 version of cudnn.

# Prepare Data
1. Download the dataset and put the dataset in ***raw_data*** file
2. Run the data preparation script

    cd train

    ./prepare.sh

# Train Model
Run the training script

./run.sh 


# Inference
Run the inference script

./infer.sh 


# References

Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Full text available at: https://arxiv.org/abs/1706.03762

Code availabel at: https://github.com/tensorflow/tensor2tensor
