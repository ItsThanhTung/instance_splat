#!/bin/bash
CODE=/lustre/scratch/client/vinai/users/tungdt33/CVPR2025/code/instance_splat
cd $CODE

casename="teatime"
dataset_path=$CODE/dataset/lerf_ovs/$casename
out_path=$dataset_path/output/$casename

echo "$casename"
echo "$dataset_path"

python train.py -s $dataset_path --model_path $out_path

casename="waldo_kitchen"
dataset_path=$CODE/dataset/lerf_ovs/$casename
out_path=$dataset_path/output/$casename

echo "$casename"
echo "$dataset_path"

python train.py -s $dataset_path --model_path $out_path
