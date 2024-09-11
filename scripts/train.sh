#!/bin/bash
CODE=/lustre/scratch/client/vinai/users/tungdt33/CVPR2025/code/instance_splat
cd $CODE

casename="figurines"
dataset_path=$CODE/dataset/lerf_ovs/$casename
echo "$casename"
echo "$dataset_path"

cd autoencoder
python train.py --dataset_path $dataset_path --segment_type sam --dataset_name $casename --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007
# python test.py --dataset_path $dataset_path/sam --dataset_name $casename
cd ../