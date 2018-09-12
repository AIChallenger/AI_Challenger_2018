#! /bin/sh

#Download the dataset and put the dataset in ../raw_data file

DATA_DIR=./t2t_data
TMP_DIR=../raw_data
mkdir -p $DATA_DIR

awk -F '\t' '{print $3}' $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/ai_challenger_MTEnglishtoChinese_trainingset_20180821.txt > $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.en
awk -F '\t' '{print $4}' $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/ai_challenger_MTEnglishtoChinese_trainingset_20180821.txt > $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.zh

#unwrap xml for valid data and test data
python prepare_data/unwrap_xml.py $TMP_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180821/ai_challenger_MTEnglishtoChinese_validationset_20180821_zh.sgm >$DATA_DIR/valid.en-zh.zh
python prepare_data/unwrap_xml.py $TMP_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180821/ai_challenger_MTEnglishtoChinese_validationset_20180821_en.sgm >$DATA_DIR/valid.en-zh.en

#Prepare Data

##Chinese words segmentation
python prepare_data/jieba_cws.py $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.zh > $DATA_DIR/wmt_enzh_32768k_tok_train.lang1
python prepare_data/jieba_cws.py $DATA_DIR/valid.en-zh.zh > $DATA_DIR/wmt_enzh_32768k_tok_dev.lang2
## Tokenize and Lowercase English training data
cat $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/wmt_enzh_32768k_tok_train.lang2
cat $DATA_DIR/valid.en-zh.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/wmt_enzh_32768k_tok_dev.lang1
