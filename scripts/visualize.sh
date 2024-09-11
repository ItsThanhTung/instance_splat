#!/bin/bash
CODE=/lustre/scratch/client/vinai/users/tungdt33/CVPR2025/code/instance_splat
cd $CODE

casename="waldo_kitchen"
dataset_path=$CODE/dataset/lerf_ovs/$casename
echo "$casename"
echo "$dataset_path"

# python visualization/visualize_gs.py --checkpoint_path $dataset_path/output/$casename/chkpnt30000.pth
cd visualization
python visualize_gs.py --checkpoint_path /lustre/scratch/client/vinai/users/tungdt33/CVPR2025/code/LangSplat/output_sam2/figurines_3/chkpnt30000.pth \
                            --ae_checkpoint_path ../autoencoder/ckpt/figurines/best_ckpt.pth
