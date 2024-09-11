import torch
import numpy as np
import random
import os

import pyviz3d.visualizer as viz
import random
from os.path import join
import open3d as o3d
import argparse
import sys
sys.path.append("..")
from autoencoder.model import Autoencoder
from utils.sh_utils import SH2RGB
from sklearn.cluster import DBSCAN
from openclip_encoder import OpenCLIPNetwork

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append([red, green, blue])
    return palette

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def read_pointcloud(pcd_path):
    scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
    point = np.array(scene_pcd.points)
    color = np.array(scene_pcd.colors)

    return point, color


class Visualization3DGS:
    def __init__(self, point, color):
        self.point = point
        self.color = color
        self.vis = viz.Visualizer()
        self.vis.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)

    def add_points(self, point, color, name):
        self.vis.add_points(name, point, color.astype(np.float32), point_size=20, visible=True)

    def save(self, path):
        self.vis.save(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        spp = torch.from_numpy(torch.load(spp_path)).to(device='cuda')
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        # skip -1 
        tt_col = self.color.copy()
        for i in range(0, uniqueness.shape[0]):
            ss = torch.where(spp == uniqueness[i].item())[0]
            for ind in ss:
                tt_col[ind,:] = pallete[int(uniqueness[i].item())]
        self.vis.add_points(f'superpoint: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')
    
    def gtviz(self, gt_data, specific = False):
        print('...Visualizing Groundtruth...')
        normalized_point, normalized_color, sem_label, ins_label = torch.load(gt_data)
        pallete =  generate_palette(int(2e3 + 1))
        n_label = np.unique(ins_label)
        tt_col = self.color.copy()
        for i in range(0, n_label.shape[0]):
            if sem_label[np.where(ins_label==n_label[i])][0] == 0 or sem_label[np.where(ins_label==n_label[i])][0] == 1: # Ignore wall/floor
                continue
            tt_col[np.where(ins_label==n_label[i])] = pallete[i]
            if specific: # be more specific
                tt_col_specific = self.color.copy()
                tt_col_specific[np.where(ins_label==n_label[i])] = pallete[i]
                self.vis.add_points(f'GT instance: ' + str(i) + '_' + class_names[sem_label[np.where(ins_label==n_label[i])][0] - 2], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'GT instance: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask3d(self, mask3d_path, specific = False):
        print('...Visualizing 3D backbone mask...')
        dic = torch.load(mask3d_path)
        instance = dic['ins']
        try:
            instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        except:
            pass
        conf3d = dic['conf']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]), self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'3D backbone mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask2d(self, mask2d_path, specific = False):
        print('...Visualizing 2D lifted mask...')
        dic = torch.load(mask2d_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        pallete =  generate_palette(int(5e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'2D lifted mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')        
        
    def finalviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing final class agnostic mask...')
        dic = torch.load(agnostic_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 5
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + class_names[label[i]], self.point, tt_col_specific, point_size=20, visible=True)                
                else:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'final mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')  

    def featureviz(self, feature_path):
        print('...Visualizing final class agnostic mask...')
        # breakpoint()
        dic = torch.load(feature_path)['feat']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        feat = torch.mean(dic, dim = -1)
        feat = feat - torch.min(feat).item()
        feat*=1000
        breakpoint()
        feat = torch.nn.functional.normalize(feat, dim = -1)
        for i in range(self.point.shape[0]):
            tt_col = tt_col[i, :]*feat[i].item()

        self.vis.add_points(f'feature: _', self.point, tt_col, point_size=20, visible=True)                
        print('---Done---')  

if __name__ == "__main__":
    
    '''
        Visualization using PyViz3D
        1. superpoint visualization
        2. ground-truth annotation
        3. 3D backbone mask (isbnet, mask3d) -- class-agnostic
        4. lifted 2D masks -- class-agnostic
        5. final masks --class-agnostic (2D+3D)
        
    
    '''
    # Scene ID to visualize
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--ae_checkpoint_path', type=str, default=None, required=False)
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
    checkpoint_path = args.checkpoint_path
    ae_checkpoint_path = args.ae_checkpoint_path

    data = torch.load(checkpoint_path)[0]
    
    if len(data) == 13: # 这是一个feature训练时保存的ckpt
        active_sh_degree,  _xyz,  _features_dc, _features_rest, \
        _scaling, _rotation, _opacity, _language_feature, max_radii2D, xyz_gradient_accum, \
        denom, opt_dict, spatial_lr_scale = data
    elif len(data) == 12: # 这是一个不训练feature保存的ckpt
        active_sh_degree,  _xyz,  _features_dc, _features_rest, \
        _scaling, _rotation, _opacity, max_radii2D, xyz_gradient_accum, \
        denom, opt_dict, spatial_lr_scale = data

    def sh_degree_to_rgb(xyz, feature):
        scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        point = np.array(scene_pcd.points)
        color = np.array(scene_pcd.colors)

        return point, color
    
    _opacity = torch.sigmoid(_opacity.detach().squeeze(-1)) > 0.0
    # color = sh_degree_to_rgb(_features_dc, _features_rest)
    point = _xyz[_opacity.squeeze(-1)].detach().cpu().numpy()
    color = SH2RGB(_features_dc[_opacity]).squeeze(1).detach().cpu().numpy()
    color = np.clip(color, 0, 1) * 255.0

    

    # Fit DBSCAN
    # dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="cosine")
    # labels = dbscan.fit_predict(_language_feature[:100000])
    _language_feature = _language_feature[_opacity].detach()
    _language_feature = _language_feature/(_language_feature.norm(dim=-1, keepdim=True) + 1e-9)   
    language_feature  = _language_feature.cpu().numpy()

    
    VIZ = Visualization3DGS(point, color)    
    
    language_feature = (language_feature * 0.5 + 0.5) * 255.0
    # VIZ.add_points(point, language_feature, "feat")

    if ae_checkpoint_path is not None:
        ae_checkpoint = torch.load(ae_checkpoint_path, map_location="cuda")
        model = Autoencoder(args.encoder_dims, args.decoder_dims).to("cuda")
        model.load_state_dict(ae_checkpoint)
        model.eval()

        clip_model = OpenCLIPNetwork("cuda")
        class_list = ["toy elephant", "red toy chair", "table"]

        with torch.no_grad():
            restored_feat = model.decode(_language_feature)
            clip_model.set_positives(class_list)
            valid_map = clip_model.get_max_across_embs(restored_feat) 
        
        for idx, class_name in enumerate(class_list):
            relevancy_embs = valid_map[idx]
            mask = (relevancy_embs > 0.6).cpu().squeeze(-1).numpy()
            VIZ.add_points(point[mask], color[mask], class_name)

    # epsilon = 10  # Distance threshold
    # min_samples = 10  # Minimum number of points to form a cluster
    # dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="cosine")
    # print(language_feature.shape)
    # labels = dbscan.fit_predict(language_feature)
    
    # cluster_mask = (labels != -1)
    # outlier_mask = (labels == -1)

    # cluster_label = labels[cluster_mask]
    # print(np.unique(cluster_label, return_counts=True))
    # color_palette = generate_palette(len(np.unique(cluster_label)))

    # VIZ.add_points(point[outlier_mask], np.zeros_like(point[outlier_mask]), "outlier")

    # cluster_color = np.array([color_palette[label] for label in cluster_label])
    # VIZ.add_points(point[cluster_mask], cluster_color, "cluster")

    # if check_superpointviz:
    #     VIZ.superpointviz(spp_path)
    # if check_gtviz:
    #     VIZ.gtviz(gt_path, specific = False)
    # if check_3dviz:
    #     VIZ.vizmask3d(mask3d_path, specific = False)
    # if check_2dviz:
    #     VIZ.vizmask2d(mask2d_path, specific = False)
    # if check_finalviz:
    #     VIZ.finalviz(agnostic_path, specific = False, vocab = False)
    # if check_featureviz:
    #     VIZ.featureviz(feature_path)
    VIZ.save("viz")