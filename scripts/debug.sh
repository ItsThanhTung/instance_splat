#!/bin/bash
CODE=/lustre/scratch/client/vinai/users/tungdt33/CVPR2025/code/instance_splat
cd $CODE

casename="teatime"
dataset_path=$CODE/dataset/lerf_ovs/$casename
echo "$casename"
echo "$dataset_path"

python preprocess_sam.py --dataset_path $dataset_path
