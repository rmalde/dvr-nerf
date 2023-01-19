import torch.optim as optim
import torch.nn as nn
import torch
from models.periodic_vanilla import PeriodicVanilla
from models.polynomial_vanilla import PolynomialVanilla

from tqdm.rich import tqdm
import numpy as np
import open3d as o3d

from utils.utils import get_device


class PeriodicTrainer:
    def __init__(self,
                 num_epochs=3,
                 in_dim=100,
                 batch_size=64,
                 epoch_len=1000):
        self.num_epochs = num_epochs
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.device = get_device()

    def train(self):
        # model = PeriodicVanilla(number_of_geometric_fn=50, in_dim=self.in_dim)
        model = PolynomialVanilla(number_of_coeffs=50, in_dim=self.in_dim)
        print(model)
        model.to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        points = np.load('data/snorlax/points.npy') # (N, 3)
        points = torch.tensor(points, device=self.device)
        min_bounds = [5.67246323e+01, -3.36619873e+01,  1.09863281e-03]
        max_bounds = [1.36311371e+02,  3.17942009e+01,  7.16124268e+01]
        for i in range(3):
            points[:, i] = (points[:, i] - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
        # [-1, 1]
        points = (points*2) - 1

        contains = np.load('data/snorlax/contains.npy') # (N,)
        contains = torch.tensor(contains, dtype=torch.float64, requires_grad=True, device=self.device)
        # pcd = o3d.geometry.PointCloud()
        # points = points.cpu().detach().numpy()
        # contains = contains.cpu().detach().numpy() 
        # pcd.points = o3d.utility.Vector3dVector(points[contains == 1])
        # o3d.io.write_point_cloud("workspace/CONTAINS.ply", pcd)

        initial = torch.rand(1, self.in_dim, device=self.device)
        # initial = torch.tile(initial, (self.batch_size, 1))

        for _ in range(self.num_epochs):
            total_loss = 0
            pbar = tqdm(range(self.epoch_len))
            for _ in range(self.epoch_len):
                #inputs, labels = data
                optimizer.zero_grad()
                coeffs = model.forward(initial)
                # preds = torch.mean(coeffs) * torch.ones_like(contains)
                preds = model.decode_vec(points, coeffs) # (N, )
                loss = self.loss_fn(contains, preds)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_description(f"Loss: {loss.item()}, output mean: {torch.mean(preds)}, output sum: {torch.sum(preds)}")
                pbar.update()
                # print(f" Inside Epoch Loss: {loss.item()}")
            pbar.close()
        print(f"Epoch Loss: {total_loss/self.epoch_len}")
        model.get_mesh(coeffs, resolution=50)
        print("about to save")
        print("Oh well i saved")
    
    def loss_fn(self, contains, preds):
        return torch.sum(torch.abs(contains - preds)) / len(preds)


if __name__ == "__main__":
    trainer = PeriodicTrainer()
    trainer.train()
