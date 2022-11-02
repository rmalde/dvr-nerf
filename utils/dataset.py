import os
import cv2
import json
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.utils import get_rays

# This file implements datasets
# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


class NeRFDataset:
    def __init__(self, data_dirs, args, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.data_dirs = data_dirs
        self.args = args
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.preload = args.preload # preload data into GPU
        self.scale = args.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = args.offset # camera offset
        self.bound = args.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = args.fp16 # if preload, load into fp16.
        
        self.num_rays = self.args.num_rays if self.type == 'train' else -1

        json_name = f"transforms_{self.type}.json"
        with open(os.path.join(self.data_dirs.colmap_dir, json_name), 'r') as f:
            transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            raise RuntimeError(f"Could not find h and w in {json_name}!")
        
        # read images
        frames = transform["frames"]
        
        self.poses = []
        self.images = []
        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            f_path = os.path.join(self.data_dirs.media_dir, f['file_path'])
            
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
            image = image.astype(np.float32) / 255 # [H, W, 3/4]

            self.poses.append(pose)
            self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]


        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.args.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)

        # load intrinsics
        fl_x = transform['fl_x'] / downscale
        fl_y = transform['fl_y'] / downscale

        cx = transform['cx'] / downscale
        cy = transform['cy'] / downscale
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index) # a list of length 1

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        # error_map = None if self.error_map is None else self.error_map[index]
        error_map = None
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.type == 'train':
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=(self.type=='train'), num_workers=0)
        # Note: this is the way that torch-ngp does their dataset,
        #       eventually we should change to just deliver
        #       the data properly rather than doing this method
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader