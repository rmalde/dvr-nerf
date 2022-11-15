import argparse
import torch 
import os
import numpy as np

import mcubes
import trimesh

from packaging import version as pver

from utils.ioutils import DataDirs
from utils.utils import custom_meshgrid, get_device, extract_geometry



def parse_args():
    parser = argparse.ArgumentParser(description="Given a trained nerf model, create a mesh")
    
    parser.add_argument('--mesher_type', type=str, default='marching_cubes', choices=['marching_cubes','mobile_nerf'], help="Which meshing_algorithm to mesh with")
    parser.add_argument('--model_checkpoint', type=str, default='', help="Which meshing_algorithm to mesh with")
    parser.add_argument('--mesh_save_path', type=str, default='workspace/mesh.obj')

    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=10)

    args = parser.parse_args()
    return args

class BaseMesher:
    def __init__(self, model, save_path, resolution=256, threshold=10):
        self.save_path = save_path
        self.model = model

        self.resolution = resolution
        self.threshold = threshold
        self.fp16 = False
        self.device = get_device()

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def mesh(self):
        raise NotImplementedError


class MarchingCubesMesher(BaseMesher):
    def query_func(self, pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma
    
    def mesh(self):
        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=self.resolution, threshold=self.threshold, query_func=self.query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(self.save_path)
        print(f"==> Finished saving mesh to {self.save_path}")
    
class MobileNerfMesher(BaseMesher):
    def mesh(self):
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()

    # data_dirs = DataDirs(args.data_dir)
    model = torch.load(args.model_checkpoint)

    masher = None
    if args.mesher_type == 'marching_cubes':
        mesher = MarchingCubesMesher(model, args.mesh_save_path, resolution=args.resolution, threshold=args.threshold)
    elif args.mesher_type == 'mobile_nerf':
        raise NotImplementedError
        # mesher = MobileNerfMesher()
    else:
        raise NotImplementedError

    mesher.mesh()