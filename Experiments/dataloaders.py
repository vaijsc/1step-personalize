from pathlib import Path

from torch.utils.data import Dataset

import numpy as np
import torch
import pdb
import os
import os.path as osp
import json
import random
import glob

def get_dict_data(task):
    if task == "coca5k":
        path_json_index = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset/filter_rgb_info_img_cap.json"
        return get_coca_personalize_task(path_json_index)
    elif task == "coca100k":
        path_json_index = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset_large/unique_info.json"
        return get_coca_personalize_task(path_json_index)
    
def get_coca_personalize_task(path_json_index, ratio_train=0.9, with_latent=False):
    with open(path_json_index, "r") as fp:
        json_data = json.load(fp)
        
    # full set
    split_train = int(ratio_train*len(json_data))
    json_train_samples = json_data[:split_train]
    
    json_val_samples = json_data[split_train:]
    
    def process_res(json_samples):
        res = {
            "ref_image": [],
            "out_image": [],
            "text": []
        }
        
        if with_latent:
            res["noisy_latent"] = []
            res["vae_latent"] = []
        
        for sample in json_samples:
            res["ref_image"].append(sample["img_path"])
            res["out_image"].append(sample["img_path"])
            res["text"].append(sample["src_p"])
            
            if with_latent:
                res["noisy_latent"].append(torch.load(sample["noisy_latent_path"]))
                res["vae_latent"].append(torch.load(sample["vae_latent_path"]))
            
        return res
    
    res_train = process_res(json_train_samples)
    res_val = process_res(json_val_samples)
    
    return res_train, res_val