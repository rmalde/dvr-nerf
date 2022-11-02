import argparse
import torch
import torch.optim as optim
import numpy as np

from utils.dataset import NeRFDataset
from utils.ioutils import DataDirs

def parse_args():
    parser = argparse.ArgumentParser(description="Train a nerf on a dataset that has been run through colmap")
    
    parser.add_argument('--model_type', type=str, default='tensorf', choices=['tensorf','tensorf_cp'], help="Which model type to train with")
    #TODO: Change to a better system for logging and chkpting
    parser.add_argument('--workspace', type=str, default='workspace')


    # ### dataset options
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset, for example data/BBB86")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--downscale', type=int, default=1, help="factor by which to downscale all images")
    
    # model options
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

    ### training options
    parser.add_argument('--lr0', type=float, default=2e-2, help="initial learning rate for embeddings")
    parser.add_argument('--lr1', type=float, default=1e-3, help="initial learning rate for networks")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--l1_reg_weight', type=float, default=4e-5)

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    print(args)

    print("Importing and building necessary libraries..")
    if args.model_type == 'tensorf':
        from models.tensorf import NeRFNetwork
        from trainers.tensorf_trainer import TensorfTrainer as Trainer
    else:
        raise NotImplementedError(f"Haven't implemented {args.model_type} model type")
    
    model = NeRFNetwork(
        resolution=[args.resolution0] * 3,
        bound=args.bound,
        cuda_ray=args.cuda_ray,
        density_scale=1,
    )

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = lambda model: torch.optim.Adam(model.get_params(args.lr0, args.lr1), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.33)
    trainer = Trainer('ngp', args, model, device=device, workspace=args.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=args.fp16, lr_scheduler=scheduler, metrics=[], use_checkpoint=args.ckpt, eval_interval=50)

    # calc upsample target resolutions - not sure what this does
    upsample_resolutions = (np.round(np.exp(np.linspace(np.log(args.resolution0), np.log(args.resolution1), len(args.upsample_model_steps) + 1)))).astype(np.int32).tolist()[1:]
    print('upsample_resolutions:', upsample_resolutions)
    trainer.upsample_resolutions = upsample_resolutions

    data_dirs = DataDirs(args.data_dir)
    train_loader = NeRFDataset(data_dirs, args, device=device, type='train').dataloader()
    valid_loader = NeRFDataset(data_dirs, args, device=device, type='val', downscale=2).dataloader()
    print("Starting training: =============================")
    trainer.train(train_loader, valid_loader, 300)
    print("Done training ================================")
    trainer.save_mesh(resolution=256, threshold=0.1)
