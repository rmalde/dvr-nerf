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
                 hidden_dim=256,
                 in_dim=6,
                 number_of_geometric_fn=100,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution

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
            nn.Linear(hidden_dim, 6*number_of_geometric_fn+1)
        )

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.encoder = nn.Sequential(*layers)
        self.mesh = None
        self.scene = None
        
        self.r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1], point_size=1.0)

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
        N = x.shape[0]
        # N, fn_coefs
        func_coefs = self.encoder(torch.cat((x, d), dim=1))
        colors, depths = torch.zeros((n, self.resolution[0], self.resolution[1], 3)), torch.empty_like((n, self.resolution[0], self.resolution[1], 1))
        for n in range(N):
            self.mesh = self.decoder(func_coefs)
            mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
            self.scene = pyrender.Scene(ambient_light= [0.3,0.3,0.3, 1.0])
            self.scene.add(mesh)
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)

            # https://gamedev.stackexchange.com/questions/45298/convert-orientation-vec3-to-a-rotation-matrix
            c1 = np.sqrt(d[n, 0]**2 + d[n, 1]**2)
            s1 =d[n, 2]

            c2 = d[n, 0] / c1
            s2 = d[n, 1] / c1
            cam_pose = np.array([
                [d[n, 0], -s2, -s1*c2, x[n, 0]],
                [d[n, 1], c2, -s1*s2, x[n, 1]],
                [d[n, 2],  0,     c1, x[n, 2]],
                [      0,  0,      0,       1],
            ])
            self.scene.add(camera, pose=cam_pose)
            color, depth = self.r.render(self.scene)
            colors[n] = color
            depths[n] = depth
            self.scene.remove_node(camera)

        return colors, depths


    def get_sigma_feat(self, x):
        # x: [N, 3], in [-1, 1]
        self.scene
        N = x.shape[0]

        # line basis
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        vec_feat = F.grid_sample(self.sigma_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.sigma_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.sigma_vec[2], vec_coord[[2]], align_corners=True).view(-1, N) # [R, N]

        sigma_feat = torch.sum(vec_feat, dim=0)

        return sigma_feat


    def get_color_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        mat_feat, vec_feat = [], []

        for i in range(len(self.color_mat)):
            mat_feat.append(F.grid_sample(self.color_mat[i], mat_coord[[i]], align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
            vec_feat.append(F.grid_sample(self.color_vec[i], vec_coord[[i]], align_corners=True).view(-1, N)) # [R, N]
        
        mat_feat = torch.cat(mat_feat, dim=0) # [3 * R, N]
        vec_feat = torch.cat(vec_feat, dim=0) # [3 * R, N]

        color_feat = self.basis_mat((mat_feat * vec_feat).T) # [N, 3R] --> [N, color_feat_dim]

        return color_feat

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

    @torch.no_grad()
    def upsample_model(self, resolution):
        self.resolution = resolution
    

    # optimizer utils
    def get_params(self, lr1, lr2):
        return [
            {'params': self.encoder.parameters(), 'lr': lr1}, 
        ]