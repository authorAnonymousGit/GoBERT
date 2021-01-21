#!/bin/bash

CONFIGURATIONS_NUM=$1;
for (( iter=1; iter<=CONFIGURATIONS_NUM; iter++ ))
do
  echo $iter;
  python -W ignore main.py \
                  --iter_num $iter \
                  --task_type "train_primary" \
                  --sub_nn $2

  python -W ignore main.py \
                  --iter_num $iter \
                  --task_type "train_sub" \
                  --sub_nn $2

done