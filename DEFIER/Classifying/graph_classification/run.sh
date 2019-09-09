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
num_epochs=5
learning_rate=0.001
fold=3

num_class=5

        # './data/badset_ori_group/badset_ori_badRan_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_badRan_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_overflow_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_overflow_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_reentran_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_reentran_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_dos_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_dos_extend_control_game_8.csv',
trainset="['./data/extend_control_game_no_transfer/goodset_4_new_extend_control_game_5.csv', \
'./data/extend_control_game_no_transfer/goodset_4_new_extend_control_game_8.csv', \
'./data/extend_control_game_no_transfer/goodset_4_new_extend_control_game_10.csv', \
'./data/extend_control_game_no_transfer/badset_ori_notransfer_extend_control_game_5.csv', \
'./data/extend_control_game_no_transfer/badset_ori_notransfer_extend_control_game_8.csv', \
'./data/extend_control_game_no_transfer/badset_ori_notransfer_extend_control_game_10.csv']"
testset="['./data/testset/class_5_26game.csv']"

        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_10.csv',
        
        #'./data/testset/class_1_20game.csv',
        #'./data/testset/class_2_18game.csv',
        #'./data/testset/class_3_5game.csv',
        # './data/testset/class_4_1600_1game.csv',
        # './data/testset/class_4_1600else_1game.csv',
        #'./data/testset/class_6_34game.csv',
        
isvalidate=1

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
    -trainset $trainset \
    -testset $testset \
    -isvalidate $isvalidate \
    $@
