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

class SinActivation(nn.Module):
    def forward(self, x):
        return x + torch.square(torch.sin(x))

class PeriodicVanilla(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 in_dim=100,
                 number_of_geometric_fn=100):
        super().__init__()

        self.in_dim = in_dim
        self.number_of_geometric_fn = number_of_geometric_fn
        self.device = get_device()

        layers = [
            nn.Linear(self.in_dim, hidden_dim),
            # nn.ReLU(),
            SinActivation()
        ]
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU()
                SinActivation()
            ]
        layers.append(
            # Each geometric fn has x_1 * sin(x theta) + x_2 * cos(x phi) +
            # x_1 * sin(y theta) + x_2 * cos(y phi) +
            # x_1 * sin(z theta) + x_2 * cos(z phi)
            nn.Linear(hidden_dim, 6*number_of_geometric_fn)
        )

        self.encoder = nn.Sequential(*layers)
        self.encoder.to(self.device)
        # 2 times the space
        self.frequencies = torch.arange(
            self.number_of_geometric_fn, device=self.device) * 2 * np.pi / self.number_of_geometric_fn
        self.frequencies = torch.unsqueeze(self.frequencies, dim=0)

    def forward(self, x):
        return self.encoder(x)

    def _evalfn(self, pos, coeff, frequency):
        frequency = 2 * np.pi * frequency / self.number_of_geometric_fn
        return (
            sum(coeff[0:3] * torch.sin(pos*frequency)) +
            sum(coeff[3:6] * torch.cos(pos*frequency))
        )

    def _eval_all_fns(self, pos, coeffs):
        total = 0
        for i, coeff in enumerate(coeffs):
            total += self._evalfn(pos, coeff, i)
        return float(total > 0)

    def decode_vec(self, positions, coeffs):
        coeffs = coeffs.reshape((self.number_of_geometric_fn, 6))

        positions = torch.unsqueeze(positions, dim=-1).float() # (n, 3, 1)
        # self.frequencies # (1, number_of_geometric_fn)
        # (n, 3, number_of_geometric_fn)
        inside_constants = positions @ self.frequencies
        outside_sin = coeffs[:, 0:3]
        outside_cos = coeffs[:, 3:6]
        # (n, 3, number_of_geometric_fn) @ (number_of_geometric_fn, 3)
        # (n, 3, 3)
        sin_outputs = torch.sin(inside_constants) @ outside_sin
        cos_outputs = outside_cos @ torch.cos(inside_constants+0.5)
        # Reduce the last two dimennsions
        output = torch.sum(sin_outputs, dim=(1, 2)) + torch.sum(cos_outputs, dim=(1, 2))
        return F.softmax(output)
        # return output 

    # def decode_vec(self, positions, coeffs):
    #     # positions: (N, 3)
    #     # coeff first half sin second half cos
    #     coeffs = coeffs.reshape((self.number_of_geometric_fn, 6)) # (num_geometric_fun , 6)
    #     # (N, 3)
    #     return torch.tensor(list(map(lambda pos: self._eval_all_fns(pos, coeffs), list(positions))), requires_grad=True, device=self.device)

    # def decode(self, pos, coeffs):
    #     coeffs = coeffs.reshape((self.number_of_geometric_fn, 6)) # (num_geometric_fun , 6)
    #     #optimize this
    #     total = 0
    #     for i, coeff in enumerate(coeffs):
    #         total += self._evalfn(pos, coeff, i)
    #     return total > 0

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
