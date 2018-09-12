bin=../tensor2tensor/bin
python $bin/t2t-trainer --registry_help

PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
HOME=`pwd`
DATA_DIR=$HOME/t2t_data
TMP_DIR=$DATA_DIR
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

# Decode

BEAM_SIZE=4
ALPHA=0.6
DECODE_FILE=$DATA_DIR/test.en

python $bin/t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=2048,shared_embedding_and_softmax_weights=0' \
  --train_steps=0 \
  --eval_steps=0 \
  --decode_beam_size=$BEAM_SIZE \
  --decode_alpha=$ALPHA \
  --decode_from_file=$DECODE_FILE

cat $DECODE_FILE.$MODEL.$HPARAMS.beam$BEAM_SIZE.alpha$ALPHA.decodes
