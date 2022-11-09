import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.encoding import get_encoder
from nerf.renderer import NeRFRenderer

import pygalmesh
import trimesh
import pyrender

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
        idx = vec_evalfn(x, y, z) < 0

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
                 hidden_dim=128,
                 number_of_geometric_fn=100,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution

        # render module (default to freq feat + freq dir)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ]
        print(hidden_layers)
        print(*hidden_layers)

        self.encoder = nn.Sequential([
            nn.Linear(6, hidden_dim), # x, y, z, [unit vector direction]
            nn.ReLU(),
            *hidden_layers,
            # Each geometric fn has x_1 * sin(x theta) + x_2 * cos(x phi) +
            # x_1 * sin(y theta) + x_2 * cos(y phi) +
            # x_1 * sin(z theta) + x_2 * cos(z phi)
            nn.Linear(hidden_dim, 6*number_of_geometric_fn+1), 
        ])
        self.mesh = None

        color_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3 # rgb
            else:
                out_dim = self.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)


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
            )
            mesh = mesher.get_mesh()
            if combined_mesh == None:
                combined_mesh = mesh
            else:
                combined_mesh = pygalmesh.Union(combined_mesh, mesh)
        # Better conversion please
        combined_mesh.write("mesh.ply")
        combined_mesh = trimesh.load('mesh.ply')

        return combined_mesh


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # normalize to [-1, 1]
        func_coefs = self.encoder(torch.cat((x, d), dim=1))
        self.mesh = self.decoder(func_coefs)

        mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        scene = pyrender.Scene(ambient_lgith = [0.3,0.3,0.3, 1.0])
        scene.add(mesh)
        pyrender.Viewer(scene)

        return self.mesh


    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        # normalize to [-1, 1]
        x = x / self.bound

        sigma_feat = self.get_sigma_feat(x)
        sigma = F.relu(sigma_feat, inplace=True)

        return {
            'sigma': sigma,
        }

    # allow masked inference
    def color(self, x, d, mask=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # normalize to [-1, 1]
        x = x / self.bound

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]

        color_feat = self.get_color_feat(x)
        color_feat = self.encoder(color_feat)
        d = self.encoder_dir(d)

        h = torch.cat([color_feat, d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs


    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_mat)):
            loss = loss + torch.mean(torch.abs(self.sigma_mat[i])) + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss

    

    # optimizer utils
    def get_params(self, lr1, lr2):
        return [
            {'params': self.encoder, 'lr': lr1}, 
        ]