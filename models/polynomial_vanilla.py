import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import pygalmesh

import open3d as o3d

from utils.utils import get_device

"""
Ideas: Greater than 0 might be a very bad way of doing it
sine cos initializing for transofmers
x + sin^2 x activation function
"""

class PolynomialVanilla(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 in_dim=100,
                 number_of_coeffs=100):
        super().__init__()

        self.in_dim = in_dim
        self.number_of_coeffs = number_of_coeffs
        self.device = get_device()

        layers = [
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()  # consider using x + sin^2(x) as function
            ]
        layers.append(
            # Each poly fn has x_1 * (x) + x_2 (y) + x_3 (z) 
            # + x_4 * (x)^2 + x_5 (y)^2 + x_6 (z)^2
            nn.Linear(hidden_dim, 3*number_of_coeffs)
        )

        self.encoder = nn.Sequential(*layers)
        self.encoder.to(self.device)

    def forward(self, x):
        return self.encoder(x)

    def decode_vec(self, positions, coeffs):
        # Each poly fn has c_1 * (x) + c_2 (y) + c_3 (z) 
        # + x_4 * (x)^2 + x_5 (y)^2 + x_6 (z)^2
        coeffs = 2 * (coeffs.reshape((self.number_of_coeffs, 3)) / torch.max(coeffs)) - 1 # (number_of_coeffs, 3)
        positions = torch.unsqueeze(positions, dim=1).float() # (n, 1, 3)
        positions = positions.repeat(1,self.number_of_coeffs,1)  # (n, num_coefficients, 3)
        # self.frequencies # (1, number_of_geometric_fn)
        # (n, 3, number_of_geometric_fn)
        powers_mask = torch.arange(0, self.number_of_coeffs).unsqueeze(0).unsqueeze(-1).to(self.device)
        positions = torch.pow(positions, powers_mask)

        return  torch.sum(positions * coeffs, dim=(1, 2))
        
        inside_constants = positions @ self.frequencies
        output = torch.sum(inside_constants, dim=(1, 2))
        return output

    def get_mesh(self, coeffs, resolution=50):
        # min_bounds = [5.67246323e+01, -3.36619873e+01,  1.09863281e-03]
        # max_bounds = [1.36311371e+02,  3.17942009e+01,  7.16124268e+01]

        min_bounds = [-1, -1, -1]
        max_bounds = [1, 1, 1]

        x_ = np.linspace(min_bounds[0], max_bounds[0], resolution)
        y_ = np.linspace(min_bounds[1], max_bounds[1], resolution)
        z_ = np.linspace(min_bounds[2], max_bounds[2], resolution)
        x, y, z = np.meshgrid(x_, y_, z_)

        # reorder so that we now have (resolution x resolution x resolution x 3)
        pos_vec = np.stack([x, y, z])  # (3, r,r,r)
        pos_vec = np.moveaxis(pos_vec, 0, -1)  # (r,r,r,3)
        pos_vec = np.reshape(pos_vec, (-1, 3))  # (r^3, 3), (N, 3)
        pos_vec = torch.tensor(pos_vec, device=self.device)
        preds = self.decode_vec(pos_vec, coeffs).int()  # (N, )
        pos_vec = pos_vec.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()        

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(pos_vec[preds == 1])

        o3d.io.write_point_cloud("workspace/TRAINING.ply", pcd)
        # mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     pcd, depth=9)
        # o3d.io.write_triangle_mesh("workspace/TRAINING.obj", mesh)
        # return mesh

        # voxel_size = (0.1, 0.1, 0.1)
        # print("about to generate mesh...")
        # mesh = pygalmesh.generate_from_array(
        #     preds, voxel_size, max_facet_distance=0.2, max_cell_circumradius=0.1
        # )
        # return mesh
