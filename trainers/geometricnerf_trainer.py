from utils.utils import get_rays, linear_to_srgb
from .trainer import Trainer

import torch
import tqdm
import os
import glob
import torch.nn.functional as F
import numpy as np


class geometricNeRFTrainer(Trainer):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        super().__init__(name, opt, model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        
    ### ------------------------------	

    def train_step(self, data):

        pred_rgb, gt_rgb, loss = super().train_step(data)

        # l1 reg
        loss += self.model.density_loss() * self.opt.l1_reg_weight

        return pred_rgb, gt_rgb, loss


    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            # data['rays_o'] = torch.nan_to_num(data['rays_o'])
            # data['rays_d'] = torch.nan_to_num(data['rays_d'])
            # data['images'] = torch.nan_to_num(data['images'])
            self.local_step += 1
            self.global_step += 1
            
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

            # Different from _Trainer!
            if self.global_step in self.opt.upsample_model_steps:

                reso = self.upsample_resolutions.pop(0)
                self.log(f"[INFO] upsample model at step {self.global_step} from {self.model.resolution[0]} to {reso}")
                self.model.upsample_model([reso] * 3)

                # reset optimizer since params changed.
                self.optimizer = self.optimizer_fn(self.model)
                self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)                

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
            'resolution': self.model.resolution, # Different from _Trainer!
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

            model_path = f"{self.ckpt_path}/{self.name}.pt"
            torch.save(self.model, model_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        # if 'model' not in checkpoint_dict:
        #     # reset resolution
        #     self.model.upsample_model() # TODO: need to calclate resolution from param size...
        #     self.optimizer = self.optimizer_fn(self.model)
        #     self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

        #     self.model.load_state_dict(checkpoint_dict)
        #     self.log("[INFO] loaded model.")
        #     return
        self.model.upsample_model(checkpoint_dict['resolution']) # Different from _Trainer!
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']

        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # mark untrained grid
            if self.global_step == 0:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
                self.error_map = False

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

            # Different from _Trainer!
            if self.global_step in self.opt.upsample_model_steps:

                # shrink
                if self.model.cuda_ray: 
                    self.model.shrink_model()

                # adaptive voxel size from aabb_train
                n_vox = self.upsample_resolutions.pop(0) ** 3 # n_voxels
                aabb = self.model.aabb_train.cpu().numpy()
                vox_size = np.cbrt(np.prod(aabb[3:] - aabb[:3]) / n_vox)
                reso = ((aabb[3:] - aabb[:3]) / vox_size).astype(np.int32).tolist()
                
                self.log(f"[INFO] upsample model at step {self.global_step} from {self.model.resolution} to {reso}")
                self.model.upsample_model(reso)

                # reset optimizer since params changed.
                self.optimizer = self.optimizer_fn(self.model)
                self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)       

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs