#!/bin/bash

DATA=TRANS

#mean_field loopy_bp
gm=mean_field

#CONV_SIZE=64
#n_hidden=64
#LV=5
LV=5
CONV_SIZE=128
FP_LEN=0
n_hidden=128
bsize=128
# num_epochs=100
num_epochs=3
learning_rate=0.001
fold=3

num_class=4

python3.6 main.py \
    -seed 1 \
    -data $DATA \
    -learning_rate $learning_rate \
    -num_epochs $num_epochs \
    -hidden $n_hidden \
    -max_lv $LV \
    -latent_dim $CONV_SIZE \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    -num_class $num_class \
    -gm $gm \
    $@
