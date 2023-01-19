import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pygalmesh
import trimesh

from utils.utils import get_device


class SinCosFn():
    def __init__(self, 
                xsin_amplitude=0,
                xcos_amplitude=0,
                ysin_amplitude=0,
                ycos_amplitude=0,
                zsin_amplitude=0,
                zcos_amplitude=0,
                frequency=0,
                period=50,
                ):
        super().__init__()
        self.xsin_amplitude = xsin_amplitude
        self.xcos_amplitude = xcos_amplitude
        self.ysin_amplitude = ysin_amplitude
        self.ycos_amplitude = ycos_amplitude
        self.zsin_amplitude = zsin_amplitude
        self.zcos_amplitude = zcos_amplitude
        self.frequency = 2 * np.pi * frequency / period

    def evalfn(self, x, y, z):
        return (
            self.xsin_amplitude * np.sin(x*self.frequency) +
            self.xcos_amplitude * np.cos(x*self.frequency) +
            self.ysin_amplitude * np.sin(y*self.frequency) +
            self.ycos_amplitude * np.cos(y*self.frequency) +
            self.zsin_amplitude * np.sin(z*self.frequency) +
            self.zcos_amplitude * np.cos(z*self.frequency)
        )

    def get_mesh(self):
        x_ = np.linspace(-1.0, 1.0, 50)
        y_ = np.linspace(-1.0, 1.0, 50)
        z_ = np.linspace(-1.0, 1.0, 50)
        x, y, z = np.meshgrid(x_, y_, z_)

        vol = np.empty((50, 50, 50), dtype=np.uint8)
        vec_evalfn = np.vectorize(self.evalfn)
        idx = vec_evalfn(x, y, z) > 0

        vol[idx] = 1
        vol[~idx] = 0

        voxel_size = (0.1, 0.1, 0.1)

        mesh = pygalmesh.generate_from_array(
            vol, voxel_size, max_facet_distance=0.2, max_cell_circumradius=0.1
        )
        return mesh

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 resolution=[128] * 3,
                 num_layers=3,
                 hidden_dim=256,
                 in_dim=3,
                 number_of_geometric_fn=100,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution
        self.device = get_device()


        # render module (default to freq feat + freq dir)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        layers = [
            nn.Linear(self.in_dim, hidden_dim), # x, y, z, [unit vector direction]
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ]
        layers.append(
            # Each geometric fn has x_1 * sin(x theta) + x_2 * cos(x phi) +
            # x_1 * sin(y theta) + x_2 * cos(y phi) +
            # x_1 * sin(z theta) + x_2 * cos(z phi)
            nn.Linear(hidden_dim, 6*number_of_geometric_fn + 1)
        )

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.encoder = nn.Sequential(*layers)
        self.mesh = None
        self.scene = None
        
        self.r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1], point_size=1.0)

    def decoder(self, gemotric_coefs):
        n = len(gemotric_coefs)
        combined_mesh = None

        # Take every 6 coefficients starting at the first (not zero) one        
        for counter, i in enumerate(range(1, gemotric_coefs, 6)):
            mesher = SinCosFn(
                xcos_amplitude=gemotric_coefs[i],
                ysin_amplitude=gemotric_coefs[i+1],
                ycos_amplitude=gemotric_coefs[i+2],
                zsin_amplitude=gemotric_coefs[i+3],
                zcos_amplitude=gemotric_coefs[i+4],
                frequency=counter,
                period=self.number_of_geometric_fn,
            )
            # TODO: Change this to out of the for loop
            mesh = mesher.get_mesh()
            if combined_mesh == None:
                combined_mesh = mesh
            else:
                combined_mesh = pygalmesh.Union(combined_mesh, mesh)
        # Better conversion please
        combined_mesh.write("mesh.ply")
        combined_mesh = trimesh.load('mesh.ply')

        return combined_mesh