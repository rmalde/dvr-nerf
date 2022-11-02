import cv2
import numpy as np
from math import log10, sqrt
import argparse

from utils.dataset import NeRFDataset
from utils.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run evlauation metrics given a model and its corresponding dataset")

    parser.add_argument('--model', type=str, required=True, help="Directory of model for example data/BBB86")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset, for example data/BBB86")
    


class Evaluator:
    def __init__(self):
        self.device = get_device()
        train_loader = NeRFDataset(data_dirs, args, device=self.device, type='train').dataloader()
    
    # extend to specific evaluators
    def run(self):
        pass

    def evaluate(self):
        pass

    def psnr(self, nerf_image, reference_image):
        mse = np.mean((reference_image - nerf_image) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def ssim(self, nerf_image, reference_image):
        return 0.5
    
    def lpips(self, nerf_image, reference_image):
        return 0.5


if __name__ == "__main__":


