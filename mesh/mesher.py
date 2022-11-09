import argparse
import torch 
import os
import numpy as np

import mcubes
import trimesh

from packaging import version as pver

from utils.ioutils import DataDirs
from utils.utils import custom_meshgrid, get_device



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
    
    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        #print('threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)

        #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
        
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles

    def extract_fields(self, bound_min, bound_max, resolution, query_func, S=128):

        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                        u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
        return u

    def query_func(self, pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma
    
    def mesh(self):
        vertices, triangles = self.extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=self.resolution, threshold=self.threshold, query_func=self.query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(self.save_path)
        f"==> Finished saving mesh to {self.save_path}"


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