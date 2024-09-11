import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import torchvision

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}/language_features"
    output_dir = f"{dataset_path}/language_features_dim3"
    os.makedirs(output_dir, exist_ok=True)
    
    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)


    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )


    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v

    for file_name in os.listdir(output_dir):
        if "f.npy" not in file_name:
            continue
    
        feat_path = os.path.join(output_dir, file_name)
        seg_path = feat_path.replace("f.npy", "s.npy")
        
        feature_map = torch.from_numpy(np.load(feat_path))
        seg_map = torch.from_numpy(np.load(seg_path))
        image_height = seg_map.size(1)
        image_width = seg_map.size(2)

        y, x = torch.meshgrid(torch.arange(0, image_height), torch.arange(0, image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1

        feature_level = 0
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask_feature = mask[0:1].reshape(1, image_height, image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask_feature = mask[1:2].reshape(1, image_height, image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask_feature = mask[2:3].reshape(1, image_height, image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask_feature = mask[3:4].reshape(1, image_height, image_width)
        else:
            raise ValueError("feature_level=", feature_level)

        # point_feature = torch.cat((point_feature2, point_feature3, point_feature4), dim=-1).to('cuda')
        point_feature = point_feature1.reshape(image_height, image_width, -1).permute(2, 0, 1)
        torchvision.utils.save_image(point_feature, feat_path.replace("_f.npy", ".png"))
        torchvision.utils.save_image(mask_feature.float(), feat_path.replace("_f.npy", "_mask.png"))
