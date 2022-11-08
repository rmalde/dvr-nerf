import os
import cv2
import numpy as np
from math import log10, sqrt
import argparse
import torch
import tqdm

from utils.dataset import NeRFDataset
from utils.utils import get_device
from utils.ioutils import DataDirs

from skimage.metrics import structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Run evlauation metrics given a model and its corresponding dataset")

    parser.add_argument('--model_checkpoint', type=str, required=True, help="path to checkpoint weights")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset, for example data/BBB86")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for dataset")
    
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--downscale', type=int, default=1, help="factor by which to downscale all images")
    # Model Options
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    #tensorf network options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--resolution0', type=int, default=128)
    parser.add_argument('--resolution1', type=int, default=300)
    parser.add_argument("--upsample_model_steps", type=int, action="append", default=[2000, 3000, 4000, 5500, 7000])
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self.data_dirs = DataDirs(args.data_dir)
        self.test_loader = NeRFDataset(self.data_dirs, args, device=self.device, type='test').dataloader()
        self.model = torch.load(args.model_checkpoint)
        self.model.eval()
        self.save_path = os.path.join(self.data_dirs.workspace, 'results')
        os.makedirs(self.save_path, exist_ok=True)

    def evaluate(self):
        psnr_scores = []
        images, preds = self.gen_images(self.test_loader)

        psnr_score = self.psnr(images, preds)
        ssim_score = self.ssim(images, preds)
        lpips_score = self.lpips(images, preds)
        return psnr_score, ssim_score, lpips_score

    def evaluate_from_files(self):
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        for i in range(25):
            path_pred = os.path.join(self.save_path, f'{i:04d}_pred.png')
            path_depth = os.path.join(self.save_path, f'{i:04d}_depth.png')
            path_gt = os.path.join(self.save_path, f'{i:04d}_gt.png')

            pred_img = cv2.imread(path_pred)
            gt_img = cv2.imread(path_gt)
            psnr_scores.append(self.psnr(pred_img, gt_img))
            ssim_scores.append(self.ssim(pred_img, gt_img))
        return np.mean(psnr_scores), np.mean(ssim_scores)



    def gen_image(self, data, i, bg_color=None, perturb=False):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images']
        H, W = data['H'], data['W']
        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.args))
        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        preds_depth = outputs['depth'].reshape(-1, H, W)
        for j, (gt_image, pred, pred_depth) in enumerate(zip(images, pred_rgb, preds_depth)):
            path_pred = os.path.join(self.save_path, f'{i:04d}_pred.png')
            path_depth = os.path.join(self.save_path, f'{i:04d}_depth.png')
            path_gt = os.path.join(self.save_path, f'{i:04d}_gt.png')
            print(f"Writing files iteration {i}")
            cv2.imwrite(path_pred, cv2.cvtColor((pred.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_depth, cv2.cvtColor((pred_depth.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_gt, cv2.cvtColor((gt_image.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return images, pred_rgb, pred_depth

    def gen_images(self, loader):
        for sample_data in loader:
            H, W, N = sample_data['images'][0].shape
            images = torch.zeros((len(loader), H, W, N))
            preds = torch.zeros((len(loader), H, W, N))
            break

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():
            if self.model.cuda_ray:                
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    self.model.update_extra_state()
            
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    image_batch, preds_batch, preds_depth = self.gen_image(data, i) 
                    images[i*self.args.batch_size:(i+1)*self.args.batch_size] = image_batch
                    preds[i*self.args.batch_size:(i+1)*self.args.batch_size] = preds_batch

                pbar.update(loader.batch_size)
                pbar.update(loader.batch_size)
        return images, preds

    def psnr(self, nerf_images, reference_images):
        # mse = np.mean((reference_images - nerf_images) ** 2, axis=(2, 3))
        mse = np.mean((reference_images - nerf_images) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def ssim(self, nerf_images, reference_images):
        # nerf_images = np.array(nerf_images).transpose((2,0,1))
        # reference_images = np.array(reference_images).transpose((2,0,1))

        # print(reference_images.shape)

        return structural_similarity(nerf_images, reference_images, channel_axis=2)
        all_ssim = []
        for output, reference in zip(nerf_images, reference_images):
            all_ssim.append(structural_similarity(output, reference))
        return np.mean(all_ssim)
    
    def lpips(self, nerf_image, reference_image):
        return 0.5


if __name__ == "__main__":
    args = parse_args()
    print(args)

    evaluator = Evaluator(args)
    # evaluator.evaluate()
    schling, bling = evaluator.evaluate_from_files()
    print(schling)
    print(bling)