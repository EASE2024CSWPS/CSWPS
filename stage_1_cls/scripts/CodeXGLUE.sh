#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir -p $1
    fi
}

SRC_DIR=..
DATA_DIR=../../data
MODEL_DIR=../../save_models/cls
SAVE_DIR=../../style_info/

make_dir $MODEL_DIR
make_dir $SAVE_DIR

DATASET=CodeXGLUE
JAVADOC_EXTENSION=original
REPO_EXTENSION=repo

make_dir $SAVE_DIR/$DATASET


function train () {

echo "============TRAINING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/javadoc.${JAVADOC_EXTENSION} \
--train_repo train/code.${REPO_EXTENSION} \
--dev_src dev/javadoc.${JAVADOC_EXTENSION} \
--dev_repo dev/code.${REPO_EXTENSION} \
--save_style False \
--save_path  ${SAVE_DIR}/${DATASET} \
--class_num 47 \
--uncase True \
--use_src_word True \
--use_src_char False \
--max_src_len 200 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 30000 \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 64 \
--num_epochs 400 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--max_relative_pos 32 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn True \
--early_stop 20 \
--warmup_steps 2000 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric acc \
--checkpoint True \
--only_test False
}




function save_style () {

echo "============SAVING STYLE============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/javadoc.${JAVADOC_EXTENSION} \
--train_repo train/code.${REPO_EXTENSION} \
--dev_src train/javadoc.${JAVADOC_EXTENSION} \
--dev_repo train/code.${REPO_EXTENSION} \
--save_style True \
--save_path  ${SAVE_DIR}/${DATASET} \
--class_num 47 \
--uncase True \
--use_src_word True \
--use_src_char False \
--max_src_len 200 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 30000 \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 64 \
--num_epochs 400 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--max_relative_pos 32 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn True \
--early_stop 20 \
--warmup_steps 2000 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric acc \
--checkpoint True \
--only_test True
}


train $1 $2
save_style $1 $2
