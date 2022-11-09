import argparse
import torch
import torch.optim as optim
import numpy as np

from utils.dataset import NeRFDataset
from utils.gui import NeRFGUI
from utils.ioutils import DataDirs
from utils.utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train a nerf on a dataset that has been run through colmap")
    
    parser.add_argument('--model_type', type=str, default='tensorf', choices=['tensorf','tensorf_cp', 'geometric'], help="Which model type to train with")
    #TODO: Change to a better system for logging and chkpting
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--test', action='store_true', help="test mode")

    # ### dataset options
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset, for example data/BBB86")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--downscale', type=int, default=1, help="factor by which to downscale all images")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for dataset")
    parser.add_argument('--max_epochs', type=int, default=100, help="Max epochs to run model")
    parser.add_argument('--gpu_number', type=int, default=0, help="GPU number to pick (first is used by display)")

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
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--l1_reg_weight', type=float, default=4e-5)

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### Experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    print(args)

    print("Importing and building necessary libraries..")
    if args.model_type == 'tensorf':
        from models.tensorf import NeRFNetwork
        from trainers.tensorf_trainer import TensorfTrainer as Trainer
    elif args.model_type == 'geometric':
        from models.geometricnerf import NeRFNetwork
        from trainers.geometricnerf_trainer import geometricNeRFTrainer as Trainer
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

    device = get_device(num=args.gpu_number)

    optimizer = lambda model: torch.optim.Adam(model.get_params(args.lr0, args.lr1), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.33)
    workspace = DataDirs(args.data_dir).workspace
    trainer = Trainer('ngp', args, model, device=device, workspace=workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=args.fp16, lr_scheduler=scheduler, metrics=[], use_checkpoint=args.ckpt, eval_interval=50)

    # calc upsample target resolutions - not sure what this does
    upsample_resolutions = (np.round(np.exp(np.linspace(np.log(args.resolution0), np.log(args.resolution1), len(args.upsample_model_steps) + 1)))).astype(np.int32).tolist()[1:]
    print('upsample_resolutions:', upsample_resolutions)
    trainer.upsample_resolutions = upsample_resolutions

    data_dirs = DataDirs(args.data_dir)
    train_loader = NeRFDataset(data_dirs, args, device=device, type='train').dataloader()
    if args.gui:
        gui = NeRFGUI(args, trainer, train_loader)
        gui.render()
    else:
        valid_loader = NeRFDataset(data_dirs, args, device=device, type='val', downscale=2).dataloader()
        print("Starting training: =============================")
        trainer.train(train_loader, valid_loader, args.max_epochs)
        print("Done training ================================")
        trainer.save_mesh(resolution=300, threshold=0.5)
