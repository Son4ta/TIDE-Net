
import torch
import os
import numpy as np
import os.path as osp
import datetime

from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip

# Import POC dataset components from our new module
from .dataset_pocflux import PocFluxDataset, gray2color as poc_gray2color, PIXEL_SCALE as poc_pixel_scale, THRESHOLDS as poc_thresholds


HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def vis_res(pred_seq, gt_seq, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=False,save_gif=False,
            pixel_scale = None, thresholds = None, gray2color = None
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = np.clip(pred_seq,a_min=0.,a_max=1.)
    gt_seq = np.clip(gt_seq,a_min=0.,a_max=1.)
    pred_seq = pred_seq.squeeze(1)
    gt_seq = gt_seq.squeeze(1)
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):
            
            # cv2.imwrite(osp.join(save_path, 'pred', f'{i}.png'), (pred * PIXEL_SCALE).astype(np.uint8))
            # cv2.imwrite(osp.join(save_path, 'targets', f'{i}.png'), (gt * PIXEL_SCALE).astype(np.uint8))
            
            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)


    if data_type=='vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)
        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)
    
    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))],dtype=np.float64)

    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)


    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2,),
    ], axis=-3)
    
    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3,)
    plt.imsave(osp.join(save_path, 'all.png'), grid_concat)
    
    if save_gif:
        clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
        clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)
    
    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask
            
        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))


DATAPATH = {
    'cikm'     : 'path/to/cikm.h5',
    'shanghai' : '/data/fcj/TIDE-Net/data/shanghai.h5',
    'meteo'    : 'path/to/meteo_radar.h5',
    'sevir'    : 'path/to/sevir2',
    'pocflux'  : '/data/fcj/ocean/code/data',
    'ecsfco2'  : '/data/fcj/TIDE-Net/data/East_China_Sea_FCO2'
}

def get_dataset(data_name, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()
    train = val = test = None
    vis_params = {} # dict to store visualization parameters

    if dataset_name == 'cikm':
        from .dataset_cikm import CIKM, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train = CIKM(DATAPATH[data_name], 'train', img_size)
        val = CIKM(DATAPATH[data_name], 'valid', img_size)
        test = CIKM(DATAPATH[data_name], 'test', img_size)
        
    elif data_name == 'shanghai':
        from .dataset_shanghai import Shanghai, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Shanghai(DATAPATH[data_name], type='train', img_size=img_size)
        val = Shanghai(DATAPATH[data_name], type='val', img_size=img_size)
        test = Shanghai(DATAPATH[data_name], type='test', img_size=img_size)
    
    elif data_name == 'meteo':
        from .dataset_meteonet import Meteo, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Meteo(DATAPATH[data_name], type='train', img_size=img_size)
        val = Meteo(DATAPATH[data_name], type='val', img_size=img_size)
        test = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        
    elif dataset_name == 'sevir':
        from .dataset_sevir import SEVIRTorchDataset, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 6, 1)#(2019, 6, 1)
        test_end_date = (2019, 12, 31)
        batch_size = kwargs.get('batch_size', 1)
        stride = kwargs.get('stride', 13)
        
        train = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=True,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size,
            num_shard=1,
            rank=0,
            start_date=None, # datetime.datetime(*(2018, 6, 1)), 
            end_date=datetime.datetime(*train_valid_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        val = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=False,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*train_valid_split),
            end_date=datetime.datetime(*valid_test_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        test = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            shuffle=False,
            img_size=img_size,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*valid_test_split),
            end_date=datetime.datetime(*test_end_date),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
    
    elif dataset_name == 'pocflux':
        # Import dataset class and visualization helpers
        from .dataset_pocflux import PocFluxDataset, gray2color, PIXEL_SCALE, THRESHOLDS
        
        # Instantiate training set and get its normalization parameters
        # Adjust other args if needed, e.g., crop_config
        train = PocFluxDataset(
            mode='train',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            # crop_config=None, # 如果想使用全图，设为None
        )
        
        # Extract normalization params computed by training set
        norm_params = {'min_val': train.min_val, 'max_val': train.max_val}
        
        # Instantiate val/test with the same normalization params
        val = PocFluxDataset(
            mode='valid',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            precomputed_norm_params=norm_params
        )
        
        test = PocFluxDataset(
            mode='test',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            precomputed_norm_params=norm_params
        )
        # Fill POC-specific visualization params
        vis_params = {
            'norm_min': train.min_val,
            'norm_max': train.max_val,
        }
        
    elif dataset_name == 'ecsfco2':
        # Import FCO2 dataloader (adjust path if needed)
        from .dataset_ECSfco2 import Fco2Dataset, gray2color_fco2, DIVERGING_CMAP

        # Fetch FCO2-specific config from kwargs or use defaults
        train_ratio = kwargs.get('train_ratio', 0.8)
        valid_ratio = kwargs.get('valid_ratio', 0.1)
        
        # 1) Initialize train set, it computes min/max for normalization
        train = Fco2Dataset(
            mode='train',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            # crop_config=crop_config,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio
        )
        
        # 2) Extract normalization params computed on train set
        norm_params = {'min_val': train.min_val, 'max_val': train.max_val}
        
        # 3) Initialize val/test with these params for consistent scaling
        val = Fco2Dataset(
            mode='valid',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            # crop_config=crop_config,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            precomputed_norm_params=norm_params
        )
        
        test = Fco2Dataset(
            mode='test',
            data_path=DATAPATH[data_name],
            seq_len=seq_len,
            img_size=img_size,
            # crop_config=crop_config,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            precomputed_norm_params=norm_params
        )
        
        # 4) Fill FCO2-specific visualization params
        vis_params = {
            'gray2color': gray2color_fco2,
            'color_map_name': DIVERGING_CMAP,
            # norm_min/norm_max are critical for de-normalizing [0,1] predictions back to physical values
            'norm_min': train.min_val,
            'norm_max': train.max_val,
        }


    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    # Fill generic visualization params for other datasets
    if dataset_name not in ['pocflux', 'ecsfco2']:
        # Dynamically import params for each dataset
        if dataset_name == 'sevir':
             from .dataset_sevir import gray2color, PIXEL_SCALE, THRESHOLDS
        # ... other datasets ...
        else: # default to cikm-like datasets
             from .dataset_cikm import gray2color, PIXEL_SCALE, THRESHOLDS

        vis_params = {
            'gray2color': gray2color,
            'pixel_scale': PIXEL_SCALE,
            'thresholds': THRESHOLDS
        }
    
    # Return dataset instances and the visualization params dict
    return train, val, test, vis_params
