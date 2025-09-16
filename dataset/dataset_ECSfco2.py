# -*- coding: utf-8 -*-
"""
PyTorch DataLoader for the processed East China Sea FCO2 HDF5 dataset (East_China_Sea_FCO2).

Aligned with PocFluxDataset in functionality and design. Key features:
- Read data from multiple year-organized HDF5 files.
- Automatically locate and sort monthly data chronologically.
- Slice consecutive monthly data into overlapping training sequences.
- Support split by ratios: 'train', 'valid', 'test'.
- Crop Region of Interest (ROI) on read to save memory and I/O.
- Compute global min/max from training data only for normalization.
- Support passing precomputed normalization parameters.
- Pad and resize arbitrary crop sizes into fixed square tensors.
- Output tensors shaped (T, C, H, W) compatible with the model.
- Include a test and visualization utility consistent with the reference style.
"""
import os
import h5py
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ====================================================================
# 1. East China Sea FCO2 dataset class (Fco2Dataset)
# ====================================================================
class Fco2Dataset(Dataset):
    """
    PyTorch dataset class for the East China Sea FCO2 HDF5 dataset.

    - Data is read from year-organized HDF5 files.
    - HDF5 internal structure: /fco2_processed/YYYYMM.
    - NaN values are handled automatically.
    - Efficient cropping, padding and resizing during loading.
    
    crop_config format: {'top': int, 'left': int, 'height': int, 'width': int}
    Example: {'top': 150, 'left': 200, 'height': 256, 'width': 256} # crop a specific ECS region
    """
    def __init__(self, mode, data_path, seq_len, img_size, 
                 crop_config={'top': 800, 'left': 300, 'height': 1024, 'width': 1024},
                 train_ratio=0.8, valid_ratio=0.1, 
                 precomputed_norm_params=None):
        """
        Initialize the dataset.
        Args:
        - mode (str): 'train', 'valid', or 'test'.
        - data_path (str): dataset root, e.g., '/data/fcj/TIDE-Net/data/East_China_Sea_FCO2/'.
        - seq_len (int): sequence length per sample.
        - img_size (int): output square image size for the model.
        - crop_config (dict, optional): crop configuration. If None, load full image.
        - train_ratio (float): train set ratio.
        - valid_ratio (float): validation set ratio.
        - precomputed_norm_params (dict, optional): precomputed normalization params {'min_val': float, 'max_val': float}.
        """
        super().__init__()
        assert mode in ['train', 'valid', 'test'], "Mode must be one of ['train', 'valid', 'test']"
        
        self.mode = mode
        self.data_path = data_path
        self.seq_len = seq_len
        self.img_size = img_size
        self.crop_config = crop_config

        # Find and sort all available monthly data
        all_monthly_data = self._find_and_sort_data()
        if not all_monthly_data:
            raise FileNotFoundError(f"No HDF5 data or 'fco2_processed' group found under {self.data_path}.")
            
        # Create samples based on sequence length
        all_samples = self._create_samples_from_data_list(all_monthly_data)

        # Split by ratios
        num_samples = len(all_samples)
        train_end = int(num_samples * train_ratio)
        valid_end = int(num_samples * (train_ratio + valid_ratio))

        if self.mode == 'train':
            self.samples = all_samples[:train_end]
        elif self.mode == 'valid':
            self.samples = all_samples[train_end:valid_end]
        else: # test
            self.samples = all_samples[valid_end:]
            print("\n=== Months contained in test samples ===")
            for idx, sample in enumerate(self.samples):
                months = [yyyymm for _, yyyymm in sample]
                print(f"Sample {idx}: {months}")
            print("=== End ===\n")

        # Compute or load normalization parameters
        if precomputed_norm_params:
            self.min_val = precomputed_norm_params['min_val']
            self.max_val = precomputed_norm_params['max_val']
            print(f"Mode '{self.mode}' loaded, using precomputed normalization. Total {len(self.samples)} samples.")
        else:
            print(f"Mode '{self.mode}' loading, computing normalization parameters...")
            # Compute normalization from training set
            train_samples_for_norm = all_samples[:train_end]
            self.min_val, self.max_val = self._calculate_normalization_params(train_samples_for_norm)
        
        if self.mode == 'train':
            print(f"Data normalization range (min, max): ({self.min_val:.4f}, {self.max_val:.4f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch one sample and return a preprocessed tensor.
        """
        # A sample consists of seq_len consecutive monthly identifiers
        sample_identifiers = self.samples[idx]
        frames = []
        
        # --- Crop on read to reduce memory/I/O ---
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
        
        for h5_path, yyyymm_key in sample_identifiers:
            try:
                with h5py.File(h5_path, 'r') as hf:
                    dset = hf[f'fco2_processed/{yyyymm_key}']
                    
                    if self.crop_config:
                        # H5Py supports slicing read efficiently
                        frame_data = dset[top:top+height, left:left+width]
                    else:
                        # No crop: read full image
                        frame_data = dset[:]

                    frames.append(frame_data.astype(np.float32))
            
            except Exception as e:
                print(f"Error reading file {h5_path} or dataset '{yyyymm_key}': {e}")
                # If file is corrupted, pad with a zero frame to keep sequence intact
                if self.crop_config:
                    error_shape = (self.crop_config['height'], self.crop_config['width'])
                else:
                    # Need full image size. Use the first valid sample as reference if available; otherwise fallback.
                    error_shape = (720, 720) # default fallback
                    if len(frames) > 0:
                        error_shape = frames[0].shape
                frames.append(np.zeros(error_shape, dtype=np.float32))

        # (T, H_crop, W_crop)
        data_array = np.stack(frames, axis=0) 
        
        # --- Preprocess ---
        # 1) Replace NaN with 0 (or other neutral value). Land mask and failed interpolation are NaN in HDF5
        data_array = np.nan_to_num(data_array, nan=0.0) 
        
        # 2) Normalize to [0, 1]
        epsilon = 1e-8 # prevent division by zero
        data_array = (data_array - self.min_val) / (self.max_val - self.min_val + epsilon)
        data_array = np.clip(data_array, 0, 1)
        
        # 3) To tensor and add channel dim: (T, C=1, H_crop, W_crop)
        data_tensor = torch.as_tensor(data_array, dtype=torch.float).unsqueeze(1)
        
        # 4) Pad to square
        _t, _c, h, w = data_tensor.shape
        if h != w:
            padding_left = (max(h, w) - w) // 2
            padding_right = max(h, w) - w - padding_left
            padding_top = (max(h, w) - h) // 2
            padding_bottom = max(h, w) - h - padding_top
            # Padding order: (left, top, right, bottom)
            data_tensor = TF.pad(data_tensor, [padding_left, padding_top, padding_right, padding_bottom], fill=0)

        # 5) Resize to target size
        data_tensor = TF.resize(data_tensor, [self.img_size, self.img_size], antialias=True)
        
        return data_tensor

    def _find_and_sort_data(self):
        """
        Scan data path, find all HDF5 files, extract all monthly dataset identifiers, sort chronologically.
        """
        search_pattern = os.path.join(self.data_path, "*.h5")
        h5_files = sorted(glob.glob(search_pattern))
        
        monthly_identifiers = []
        print(f"Found {len(h5_files)} HDF5 files, scanning contents...")
        for h5_path in tqdm(h5_files, desc="Scanning HDF5 files"):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    if 'fco2_processed' in hf:
                        # Get all month keys (e.g., '201001', '201002')
                        for yyyymm_key in sorted(hf['fco2_processed'].keys()):
                            monthly_identifiers.append((h5_path, yyyymm_key))
            except Exception as e:
                print(f"Warning: failed to read file {h5_path}: {e}")

        # Final sort by yyyymm to ensure global chronological order
        monthly_identifiers.sort(key=lambda x: x[1])
        
        print(f"Found {len(monthly_identifiers)} valid monthly entries.")
        return monthly_identifiers

    def _create_samples_from_data_list(self, data_list):
        """Create samples from identifier list given sequence length."""
        num_frames = len(data_list)
        if num_frames < self.seq_len:
            raise ValueError(f"Total frames ({num_frames}) < sequence length ({self.seq_len}); cannot create samples.")
        samples = []
        for i in range(num_frames - self.seq_len + 1):
            samples.append(data_list[i : i + self.seq_len])
        return samples
        
    def _calculate_normalization_params(self, train_samples):
        """
        Compute normalization parameters based on training samples (identifier list).
        For efficiency, perform this on the cropped ROI as well.
        """
        print("Computing global min/max on training set for normalization...")
        min_val, max_val = np.inf, -np.inf
        
        # Get unique train identifiers
        unique_train_data = sorted(list(set(item for sample in train_samples for item in sample)))
        
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
            print(f"Compute normalization only within crop ROI [T:{top}, L:{left}, H:{height}, W:{width}].")
        
        for h5_path, yyyymm_key in tqdm(unique_train_data, desc="Computing normalization params"):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    dset = hf[f'fco2_processed/{yyyymm_key}']
                    
                    if self.crop_config:
                        frame_data = dset[top:top+height, left:left+width]
                    else:
                        frame_data = dset[:]
                    
                    # Use nanmin/nanmax to ignore NaN
                    current_min = np.nanmin(frame_data)
                    current_max = np.nanmax(frame_data)
                    
                    if not np.isnan(current_min):
                        min_val = min(min_val, current_min)
                    if not np.isnan(current_max):
                        max_val = max(max_val, current_max)
                        
            except Exception as e:
                print(f"Warning: skip {h5_path}/{yyyymm_key} when computing normalization (reason: {e})")

        return min_val, max_val


# ===============================================================
# 2. Visualization parameters for FCO2 dataset
# ===============================================================

# FCO2 has positive/negative values; a diverging colormap fits.
# RdBu_r: red for positive flux (ocean → atmosphere), blue for negative (ocean uptake)
DIVERGING_CMAP = 'RdBu_r'

def gray2color_fco2(image, vmin=-5, vmax=5):
    """
    Map single-channel FCO2 grayscale to color by physical values.
    Note: image is original unnormalized physical value.
    """
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(DIVERGING_CMAP)
    colored_image = cmap(norm(image))
    return colored_image


# ===============================================================
# 3. Test function for crop visualization (server-friendly)
# ===============================================================
def test_and_visualize_crop(data_path, crop_config, output_filepath):
    """
    Visualize dataset crop region and save to an image file.
    Args:
    - data_path (str): dataset root.
    - crop_config (dict): crop config to test.
    - output_filepath (str): output image path.
    """
    print("--- Start crop visualization test (server mode) ---")

    # Find one HDF5 file as background
    try:
        dataset_for_test = Fco2Dataset(mode='test', data_path=data_path, seq_len=1, img_size=128, train_ratio=0.99, valid_ratio=0.0)
        all_data_ids = dataset_for_test._find_and_sort_data()
        if not all_data_ids:
            raise FileNotFoundError("No valid monthly data found under the specified path.")
    except Exception as e:
        print(f"Error: Failed to initialize dataset or locate files. Check path '{data_path}'. Error: {e}")
        return

    # Use first month as global background
    global_map_h5_path, global_map_key = all_data_ids[0]
    print(f"Using dataset '{global_map_key}' (from file '{os.path.basename(global_map_h5_path)}') as background map.")
    
    try:
        with h5py.File(global_map_h5_path, 'r') as hf:
            global_data = hf[f'fco2_processed/{global_map_key}'][:]
            # Load land mask for clearer background
            if 'land_mask' in hf:
                land_mask = hf['land_mask'][:]
                global_data[land_mask] = np.nan
    except Exception as e:
        print(f"Error: Failed to read HDF5 file '{global_map_h5_path}'. Error: {e}")
        return

    # If crop_config is not provided, cannot proceed
    if not crop_config:
        print("Error: crop_config not provided; cannot run visualization test.")
        return

    print(f"Loading data with crop_config: {crop_config}")
    # Create a dataset instance with this crop config
    dataset = Fco2Dataset(mode='test', data_path=data_path, seq_len=1,
                          img_size=crop_config['height'],
                          crop_config=crop_config,
                          train_ratio=0.99, valid_ratio=0.0)

    if len(dataset) == 0:
        print("Error: Dataset failed to load any samples; cannot generate image.")
        return

    # Get first sample and convert to numpy for display
    # dataset[0] output is normalized to [0,1]
    cropped_tensor_normalized = dataset[0]
    cropped_image_normalized = cropped_tensor_normalized.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'FCO2 Dataset Crop Verification: top={crop_config["top"]}, left={crop_config["left"]}', fontsize=16)

    # Subplot 1: original map and crop box
    ax1 = axes[0]
    # Use a symmetric range to show positive/negative flux
    v_abs = np.nanpercentile(np.abs(global_data), 98)
    im1 = ax1.imshow(global_data, cmap=DIVERGING_CMAP, vmin=-v_abs, vmax=v_abs)
    ax1.set_title(f'Original Data Map with Crop Area\n(Full Shape: {global_data.shape})')
    ax1.set_xlabel('Longitude Index')
    ax1.set_ylabel('Latitude Index')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='FCO2 (mol/m^2/yr)')
    
    # Draw crop box
    rect = patches.Rectangle((crop_config['left'], crop_config['top']), crop_config['width'], crop_config['height'],
                             linewidth=2.0, edgecolor='lime', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    
    # Zoom into crop area
    padding_x = crop_config['width'] * 1.5
    padding_y = crop_config['height'] * 1.5
    ax1.set_xlim(max(0, crop_config['left']-padding_x), min(global_data.shape[1], crop_config['left']+crop_config['width']+padding_x))
    ax1.set_ylim(min(global_data.shape[0], crop_config['top']+crop_config['height']+padding_y), max(0, crop_config['top']-padding_y))

    # Subplot 2: dataset output after loading/processing
    ax2 = axes[1]
    # Display normalized data
    im2 = ax2.imshow(cropped_image_normalized, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title(f'Actual Dataset Output (Cropped & Processed)\n(Final Shape: {cropped_image_normalized.shape})')
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('Pixel')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Normalized Value [0, 1]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- Key change: save file instead of showing ---
    try:
        plt.savefig(output_filepath, bbox_inches='tight', dpi=150)
        print(f"\nSuccess! Visualization saved to: {os.path.abspath(output_filepath)}")
    except Exception as e:
        print(f"\nError: Failed to save image. Check path '{output_filepath}' and write permissions. Error: {e}")
    
    plt.close(fig) # close figure and free memory
    print("--- Test finished ---")


# ===============================================================
# 4. Main entry (server-friendly)
# ===============================================================
if __name__ == '__main__':
    # --- Config ---
    
    # Ensure headless server run
    import matplotlib
    matplotlib.use('Agg') # key: use non-interactive backend

    # Important: ensure this points to your HDF5 data dir
    # Should contain files like 'East_China_Sea_FCO2_2010.h5', 'East_China_Sea_FCO2_2011.h5'
    DATA_ROOT_PATH = "/data/fcj/TIDE-Net/data/East_China_Sea_FCO2"

    # Define a crop region to test (top, left, height, width)
    # (0,0) is image top-left
    CROP_CONFIG_TO_TEST = {'top': 700, 'left': 300, 'height': 1024, 'width': 1024}
    
    # Output visualization image path
    OUTPUT_IMAGE_PATH = 'fco2_crop_visualization.png'

    # --- Run test ---
    if not os.path.isdir(DATA_ROOT_PATH):
        print(f"Error: specified data path is not a valid directory: '{DATA_ROOT_PATH}'")
        print("Please update 'DATA_ROOT_PATH' to your correct path.")
    else:
        test_and_visualize_crop(
            data_path=DATA_ROOT_PATH,
            crop_config=CROP_CONFIG_TO_TEST,
            output_filepath=OUTPUT_IMAGE_PATH
        )
        
    # --- How to use Fco2Dataset in your training script ---
    print("\n--- Fco2Dataset usage example ---")
    try:
        # 1) Initialize training set
        train_dataset = Fco2Dataset(
            mode='train',
            data_path=DATA_ROOT_PATH,
            seq_len=12,  # e.g., use 12 months to predict future
            img_size=256,
            crop_config=CROP_CONFIG_TO_TEST,
            train_ratio=0.8, # 80% train
            valid_ratio=0.1  # 10% valid -> 10% test
        )

        # 2) Get normalization params from training set for val/test
        norm_params = {
            'min_val': train_dataset.min_val,
            'max_val': train_dataset.max_val
        }
        
        # 3) Initialize validation set with same params
        valid_dataset = Fco2Dataset(
            mode='valid',
            data_path=DATA_ROOT_PATH,
            seq_len=12,
            img_size=256,
            crop_config=CROP_CONFIG_TO_TEST,
            train_ratio=0.7,
            valid_ratio=0.2,
            precomputed_norm_params=norm_params # 传入训练集的归一化参数
        )

        # 4) Create DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

        # 5) Iterate one batch for sanity check
        first_batch = next(iter(train_loader))
        print(f"\nSuccessfully created DataLoaders.")
        print(f"One batch tensor shape: {first_batch.shape}")
        print(f"Expected: (batch_size, seq_len, num_channels, height, width)")
        print(f"Actual: {first_batch.shape}")
        assert list(first_batch.shape) == [4, 12, 1, 256, 256]
        print("Shape check passed!")

    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"\nError while running example: {e}")
    except Exception as e:
        print(f"\nUnknown error: {e}")