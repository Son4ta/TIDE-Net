import os
import os.path as osp
import math
import time
import argparse
import logging 
import yaml
from tqdm import tqdm
from datetime import timedelta
import numpy as np

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from dataset.get_datasets import get_dataset
from utils.tools import print_log, cycle, show_img_info

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "2a8cf694cbbc537b035aaf60f668ec4fe0916ddf"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



from utils.visualization import visualize_and_save


def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',       type=str,   default='tidenet',        help='tidenet_diffusion tidenet backbone model for deterministic prediction')
    parser.add_argument("--seed",           type=int,   default=114514,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='basic_exps',   help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default="4-6-6-l2-pha=0",           help="additional note for experiment")
    # CUDA_VISIBLE_DEVICES=1 conda_envs/tide-net/bin/python TIDE-Net/run.py 6_[1]KLx+1
    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='ecsfco2',        help="ecsfco2 pocflux dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--stride",         type=int,   default=1,              help="dataset stride")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--patch",          type=int,   default=2,              help="patch size")
    parser.add_argument("--seq_len",        type=int,   default=12,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=6,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=6,              help="number of frames to output")    
    parser.add_argument("--num_workers",    type=int,   default=8,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=1e-5,           help="0.0 l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=200,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=8,               help="batch size")
    parser.add_argument("--epochs",         type=int,   default=100,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=1,               help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    parser.add_argument("--spec_num",       type=int,   default=20,              help="spectral number")
    parser.add_argument("--layers",         type=int,   default=3,               help="layers number")
    parser.add_argument("--pha_weight",     type=float, default=0.001,           help="0.001 ablation I (remove phase loss) phase weight")
    parser.add_argument("--amp_weight",     type=float, default=0.01,            help="0.01 ablation II (remove amplitude loss) amplitude weight")
    parser.add_argument("--anet_weight",    type=float, default=0.1,             help="0.1 ablation II (remove amplitude loss) amplitude network MSE weight")
    parser.add_argument("--aw_stop_step",   type=int,   default=1000,            help="training step at which the amplitude weight decays to 0")
    parser.add_argument("--out_weight",     type=float, default=1.0,             help="final output weight")
    parser.add_argument("--tf",             action="store_false",                help="teacher force")
    parser.add_argument("--tf_stop_iter",     type=int,     default=2000,        help="teacher force stop iters")
    parser.add_argument("--tf_changing_rate", type=float,   default=0.,          help="teacher force changing rate")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",   default=False,                help="evaluation mode")
    parser.add_argument("--valid",          action="store_true",   default=True,              help="valid mode")
    parser.add_argument("--valid_limit",    action="store_true",                 help="valid limit mode")
    parser.add_argument("--vlnum",          type=int,   default=10,              help="valid limit nums")
    parser.add_argument("--visual",         action="store_true",  default=True,  help="save all test sample visualization")
    parser.add_argument("--wandb_state",    type=str,   default='disabled',      help="online disabled wandb state config")
    parser.add_argument("--gpu_use",        type=str,   nargs='+', default=[],   help="gpu(s) to use")
    parser.add_argument("--res_opt",        action="store_true",                 help="resume opt")

    args = parser.parse_args()
    
    return args

class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        self._preparation()
        # CSI: higher is better, initialize as 0
        self.max_csi, self.best_csi_step = 0.0, 0
        # MSE/MAE: lower is better, initialize as +inf
        self.min_mse, self.best_mse_step = float('inf'), 0
        self.min_mae, self.best_mae_step = float('inf'), 0

        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        '''
        When find_unused_parameters is set to False,
        DDP assumes all parameters with requires_grad=True participate in loss computation and produce gradients.
        If a parameter does not produce a gradient, DDP will wait for the missing gradient during backward,
        which can cause the training process to hang.
        '''
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=180))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   'wandb'
        )
        
        # Config log tracker 'wandb' from accelerate
        self.accelerator.init_trackers(
            project_name=self.exp_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": 
                {
                "mode": self.args.wandb_state,
                # 'resume': self.args.ckpt_milestone
                }
                         }   # disabled, online, offline
        )
        print_log(f"Using GPUs: {self.device}", self.is_main)
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.train_loader, self.valid_loader, self.test_loader
        )
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optimizer, self.scheduler
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================

        train_data, valid_data, test_data, vis_params = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
        )
        
        # Save the whole dict for later use
        self.vis_params = vis_params
        # For compatibility with the existing Evaluator, still extract scale and thresholds
        if self.args.dataset  in ['pocflux', 'ecsfco2']:
            # PocFlux data is in [0,1]; use default settings for metrics
            self.thresholds = [0.1, 0.3, 0.5, 0.7]
            self.scale_value = 1.0
        else:
            # Other datasets fetch values from the dict
            self.thresholds = self.vis_params.get('thresholds')
            self.scale_value = self.vis_params.get('pixel_scale')
        
        if self.args.dataset != 'sevir':
            # preload big batch data for gradient accumulation
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {self.scale_value}, Threshold: {str(self.thresholds)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================
        print_log("Build Model!", self.is_main)
        if self.args.backbone == 'simvp':
            from models.simvp import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        elif self.args.backbone == 'tidenet':
            from models.tidenet import get_model
            kwargs = {
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                'img_channels' : self.args.img_channel,
                'dim' : 64,
                'n_layers': self.args.layers,
                'pha_weight': self.args.pha_weight,
                'anet_weight': self.args.anet_weight,
                'amp_weight': self.args.amp_weight,
                'spec_num': self.args.spec_num,
                'aweight_stop_steps': self.args.aw_stop_step,
            }
            model = get_model(**kwargs)
        elif self.args.backbone == 'tidenet_diffusion':
            # Import the new TIDENet_Diffusion model
            from models.tidenet_diffusion import get_model

            # Configure TIDENet_Diffusion model parameters
            kwargs = {
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "spec_num": self.args.spec_num,  # Required by AlphaMixer
                #"num_diffusion_steps": self.args.num_diffusion_steps,  # diffusion steps
            }
            model = get_model(**kwargs)
        else:
            raise NotImplementedError
            
        self.model = model
        print_log("begin ema", self.is_main)
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        print_log("end device", self.is_main)
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)

    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch)
        
        self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self, svname=None):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        if svname == None:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
            print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        else:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{svname}.pt"))
            print_log(f"Save {svname} checkpoint to {self.ckpt_path}", self.is_main)

        
    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if isinstance(milestone, str) and '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.model = self.accelerator.prepare(model)
        if self.args.res_opt:
            try:
                self.optimizer.load_state_dict(data['opt'])
                self.scheduler.load_state_dict(data['scheduler'])
            except:
                print_log(f"No optimizer", self.is_main)
            try:
                self.cur_epoch = data['epoch'] + 1 
            except:
                print_log(f"No record epoch", self.is_main)

            try:
                self.cur_step = data['step']
            except:
                print_log(f"No record step", self.is_main)
            
        if self.is_main:
            self.ema.load_state_dict(data['ema'])


    def train(self):
        # set global step as traing process
        # torch.autograd.set_detect_anomaly(True)
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps,
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                # train the model with mixed_precision
                # with self.accelerator.autocast(self.model):
                with self.accelerator.autocast():
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])
                    
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                for k,v in loss_dict.items():
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 20 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon= self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                visualize_and_save(
                                    dataset_name=self.args.dataset,
                                    # Use correct variable names: radar_ori and radar_recon
                                    # The visualization function takes entire batch tensors
                                    target=torch.from_numpy(radar_ori),
                                    prediction=torch.from_numpy(radar_recon),
                                    save_path=osp.join(self.sanity_path, f"{i}/vil"),
                                    # For sanity check, visualize only the first batch
                                    sample_index=0, 
                                    # Pass all visualization parameters via dict
                                    **self.vis_params
                                )
                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            # save checkpoint and do test every epoch
            if self.args.valid:
                res_dict = self.test_samples(self.cur_step)

                cur_csi = res_dict.get('csi', 0.0)
                cur_mse = res_dict.get('mse', float('inf'))
                cur_mae = res_dict.get('mae', float('inf'))
                if self.args.valid_limit:
                    self.save()
                else:
                    # 1) Check if CSI is the best
                    if cur_csi > self.max_csi:
                        self.max_csi = cur_csi
                        self.best_csi_step = self.cur_step
                        self.save('best_csi') # save as best_csi.pt
                        print_log(f"✅ New best CSI model found! CSI: {self.max_csi:.4f} at step {self.best_csi_step}. Checkpoint saved.", self.is_main)

                    # 2) Check if MSE is the best
                    if cur_mse < self.min_mse:
                        self.min_mse = cur_mse
                        self.best_mse_step = self.cur_step
                        self.save('best_mse') # save as best_mse.pt
                        print_log(f"✅ New best MSE model found! MSE: {self.min_mse:.4f} at step {self.best_mse_step}. Checkpoint saved.", self.is_main)
                    # 3) Check if MAE is the best
                    if cur_mae < self.min_mae:
                        self.min_mae = cur_mae
                        self.best_mae_step = self.cur_step
                        self.save('best_mae') # save as best_mae.pt
                        print_log(f"✅ New best MAE model found! MAE: {self.min_mae:.4f} at step {self.best_mae_step}. Checkpoint saved.", self.is_main)

                    self.save('last')
                    print_log(f"Validation results at step {self.cur_step}: CSI={cur_csi:.4f}, MSE={cur_mse:.4f}, MAE={cur_mae:.4f}", self.is_main)
                    print_log(f"Current bests: CSI={self.max_csi:.4f} (step {self.best_csi_step}), MSE={self.min_mse:.4f} (step {self.best_mse_step}), MAE={self.min_mae:.4f} (step {self.best_mae_step})", self.is_main)
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
            else:
                self.save()
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch[:, :self.args.frames_out + self.args.frames_in]       # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        
        # if self.args.backbone == 'tidenet_diffusion':
        #     loss = self.model(frames_in, frames_out)
        if hasattr(self.model, 'module'):
            _, loss = self.model.module.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        else:
            _, loss = self.model.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        
        if isinstance(loss, dict):
            if 'total_loss' in loss:
                return loss
            else:
                raise ValueError("The loss must contain the 'total_loss' key.")
        else:
            return {'total_loss': loss}
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False, vis_diff=False):
        # sample_fn = self.ema.ema_model.predict if use_ema else self.model.predict
        sample_fn = (self.ema.ema_model.module.predict if hasattr(self.ema.ema_model, 'module') else self.ema.ema_model.predict) if use_ema else (self.model.module.predict if hasattr(self.model, 'module') else self.model.predict)
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        # radar_pred, *_ = sample_fn(radar_input,compute_loss=False)
        radar_pred, *_ = sample_fn(radar_input, frames_gt=radar_gt)
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred
    
    
    def test_samples(self, milestone, do_test=False):
        save_vis = True
        # init test data loader
        data_loader = self.test_loader if do_test else self.valid_loader
        # init sampling method
        self.model.eval()
        # init test dir config
        cnt = 0
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        from utils.metrics import Evaluator
        eval = Evaluator(
            seq_len=self.args.frames_out,
            value_scale=self.scale_value,
            thresholds=self.thresholds,
            save_path=save_dir,
        )

        # start test loop
        valid_nums = 0
        for i, batch in enumerate(tqdm(data_loader, desc='Test Samples', disable=not self.is_main)):
            # sample
            radar_ori, radar_recon= self._sample_batch(batch)
            # evaluate result and save
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)
                should_visualize = do_test and self.args.visual
                if should_visualize:
                    visualize_and_save(
                        dataset_name=self.args.dataset,
                        # Use correct variable names radar_ori and radar_recon
                        target=torch.from_numpy(radar_ori),
                        prediction=torch.from_numpy(radar_recon),
                        save_path=osp.join(save_dir, f"{i}/vil"),
                        sample_index=i,  # Use outer loop batch index i
                        **self.vis_params
                    )

            self.accelerator.wait_for_everyone()
            valid_nums += 1
            if not do_test and self.args.valid_limit and valid_nums >= self.args.vlnum:
                break
            # cnt += 1
            # if cnt > 10:
            #     break
        # test done
        if self.is_main:
            res = eval.done(is_main_process=self.is_main)
            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("="*30)

            if self.args.valid:
                return res
        else:
            return None

        
    def check_milestones(self, target_ckpt=None):

        mils_paths = os.listdir(self.ckpt_path)
        try:
            milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        except:
            milestones = [m.split('-')[-1].split('.')[0] for m in mils_paths]
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)
        
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            return
        
        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)
        
            
def main():
    args = create_parser()
    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    exp = Runner(args)
    if not args.eval:
        exp.train()
        exp.check_milestones()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    main()