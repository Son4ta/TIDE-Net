import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from pyhdf.SD import SD, SDC
from matplotlib import colors

# ===============================================================
# 1. POC Flux 数据集类 (你提供的代码)
# ===============================================================
class PocFluxDataset(Dataset):
    """
    用于 MODIS-based POC Flux 数据集的PyTorch数据集类。
    - 从按年份/月份组织的多HDF4文件中读取数据。
    - 可选择 'poc_flux' 或 'pe_ratio' 变量。
    - 自动处理无效值 (-9999.)。
    - 自动将数据填充、缩放为正方形。
    + crop_config 格式: (top, left, height, width)
    + 例如: {'top': 500, 'left': 3600, 'height': 720, 'width': 720}, # 示例: 裁剪出太平洋东岸
    """
    def __init__(self, mode, data_path, seq_len, img_size, 
                 variable='poc_flux', # 选择要加载的变量
                 crop_config={'top': 700, 'left': 3700, 'height': 256, 'width': 256}, # 裁剪配置参数
                 train_ratio=0.8, valid_ratio=0.1, 
                 precomputed_norm_params=None):
        """
        初始化数据集。
        参数:
        - mode (str): 'train', 'valid', 或 'test'.
        - data_path (str): 数据集根目录, e.g., '/data/fcj/ocean/code/data/'.
        - seq_len (int): 每个样本的时间序列长度。
        - img_size (int): 输出给模型的正方形图像尺寸。
        - variable (str): 要加载的数据集名称, 'poc_flux' 或 'pe_ratio'.
        - train_ratio (float): 训练集比例。
        - valid_ratio (float): 验证集比例。
        - precomputed_norm_params (dict): 预计算的归一化参数 {'min_val': float, 'max_val': float}。
        """
        super().__init__()
        assert mode in ['train', 'valid', 'test'], "Mode must be one of ['train', 'valid', 'test']"
        assert variable in ['poc_flux', 'pe_ratio'], "Variable must be one of ['poc_flux', 'pe_ratio']"
        
        self.mode = mode
        self.data_path = data_path
        self.seq_len = seq_len
        self.img_size = img_size
        self.variable = variable
        self.crop_config = crop_config

        all_files = self._find_and_sort_files()
        if not all_files:
            raise FileNotFoundError(f"在路径 {self.data_path} 下没有找到任何数据文件。")
            
        all_samples = self._create_samples_from_files(all_files)

        num_samples = len(all_samples)
        train_end = int(num_samples * train_ratio)
        valid_end = int(num_samples * (train_ratio + valid_ratio))

        if self.mode == 'train':
            self.samples = all_samples[:train_end]
        elif self.mode == 'valid':
            self.samples = all_samples[train_end:valid_end]
        else: # test
            self.samples = all_samples[valid_end:]

        if precomputed_norm_params:
            self.min_val = precomputed_norm_params['min_val']
            self.max_val = precomputed_norm_params['max_val']
            print(f"模式 '{self.mode}' 加载完成，使用预计算的归一化参数。变量: '{self.variable}'。共 {len(self.samples)} 个样本。")
        else:
            print(f"模式 '{self.mode}' 加载中 (变量: '{self.variable}'), 需要计算归一化参数...")
            # 使用训练集的文件来计算归一化参数
            train_files_for_norm = all_samples[:train_end]
            self.min_val, self.max_val = self._calculate_normalization_params(train_files_for_norm)
        
        if self.mode == 'train':
            print(f"数据归一化范围 (min, max): ({self.min_val:.2f}, {self.max_val:.2f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本。
        """
        sample_files = self.samples[idx]
        frames = []
        
        # <--- 优化点 1: 在读取时裁剪，大幅降低内存占用 ---
        # 提取裁剪参数
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
        
        for file_path in sample_files:
            hdf = None # 初始化hdf变量
            try:
                hdf = SD(file_path, SDC.READ)
                sds = hdf.select(self.variable)
                
                if self.crop_config:
                    # 直接读取裁剪后的区域，而不是整个文件
                    # pyhdf select对象支持类似numpy的切片
                    frame_data = sds[top:top+height, left:left+width]
                else:
                    # 如果没有裁剪配置，则读取整个图像
                    frame_data = sds.get()

                # 处理无效值 -9999.
                frame_data = frame_data.astype(np.float32) # 确保数据类型正确
                frame_data[frame_data == -9999.] = np.nan
                frames.append(frame_data)
                
            except Exception as e:
                print(f"读取文件 {file_path} 或变量 '{self.variable}' 时出错: {e}")
                # 如果文件损坏，用一个全零的帧代替
                # 确保错误帧的尺寸与裁剪后的一致
                error_frame_shape = (self.crop_config['height'], self.crop_config['width']) if self.crop_config else (2160, 4320)
                frames.append(np.zeros(error_frame_shape, dtype=np.float32))
            finally:
                # 确保HDF文件被关闭，防止资源泄漏
                if hdf:
                    hdf.end()
        # --- (优化结束) ---

        data_array = np.stack(frames, axis=0) # (T, H_crop, W_crop)
        data_array = np.nan_to_num(data_array, nan=0.0) # 将NaN替换为0
        
        epsilon = 1e-8
        data_array = (data_array - self.min_val) / (self.max_val - self.min_val + epsilon)
        data_array = np.clip(data_array, 0, 1)
        
        # 直接在as_tensor中添加通道维度，代码更简洁
        data_tensor = torch.as_tensor(data_array, dtype=torch.float).unsqueeze(1) # (T, C=1, H_crop, W_crop)
        
        _t, _c, h, w = data_tensor.shape
        if h != w:
            padding_left = (max(h, w) - w) // 2
            padding_right = max(h, w) - w - padding_left
            padding_top = (max(h, w) - h) // 2
            padding_bottom = max(h, w) - h - padding_top
            data_tensor = TF.pad(data_tensor, [padding_left, padding_top, padding_right, padding_bottom], fill=0)

        data_tensor = TF.resize(data_tensor, [self.img_size, self.img_size], antialias=True)
        
        # **重要修改**: 为了适配框架，这里只返回数据张量
        return data_tensor

    def _find_and_sort_files(self):
        """扫描数据路径，找到所有HDF文件并按时间排序。"""
        all_files = []
        for year_folder in sorted(os.listdir(self.data_path)):
            year_path = os.path.join(self.data_path, year_folder)
            if os.path.isdir(year_path) and year_folder.startswith("GlobalMarinePOC_"):
                for file_name in sorted(os.listdir(year_path)):
                    if file_name.endswith(".hdf"):
                        all_files.append(os.path.join(year_path, file_name))
        return all_files

    def _create_samples_from_files(self, files):
        """根据文件列表和序列长度创建样本。"""
        num_frames = len(files)
        if num_frames < self.seq_len:
            raise ValueError(f"数据总帧数({num_frames})小于序列长度({self.seq_len})，无法创建样本。")
        samples = []
        for i in range(num_frames - self.seq_len + 1):
            samples.append(files[i : i + self.seq_len])
        return samples
        
    def _calculate_normalization_params(self, train_samples):
        """
        基于训练集样本（文件列表）计算归一化参数。
        """
        print("正在基于训练集计算数据的全局最大值和最小值用于归一化...")
        min_val, max_val = np.inf, -np.inf
        
        unique_train_files = sorted(list(set(file for sample in train_samples for file in sample)))
        
        # <--- 优化点 2: 在计算归一化时也只读取裁剪区域 ---
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
            print(f"将仅在裁剪区域 [T:{top}, L:{left}, H:{height}, W:{width}] 内计算归一化参数。")
        
        for file_path in tqdm(unique_train_files, desc="计算归一化参数"):
            hdf = None
            try:
                hdf = SD(file_path, SDC.READ)
                sds = hdf.select(self.variable)
                
                if self.crop_config:
                    # 只读取裁剪区域的数据进行计算
                    frame_data = sds[top:top+height, left:left+width]
                else:
                    frame_data = sds.get()
                
                frame_data = frame_data.astype(np.float32)
                frame_data[frame_data == -9999.] = np.nan
                
                current_min = np.nanmin(frame_data)
                current_max = np.nanmax(frame_data)
                
                if not np.isnan(current_min):
                    min_val = min(min_val, current_min)
                if not np.isnan(current_max):
                    max_val = max(max_val, current_max)
                    
            except Exception as e:
                print(f"警告: 在计算归一化时跳过文件 {file_path} (原因: {e})")
            finally:
                if hdf:
                    hdf.end()
        # --- (优化结束) ---

        return min_val, max_val


# ===============================================================
# 2. 为 POC Flux 数据集定义可视化参数
# ===============================================================

# 像素值缩放，因为你的数据已经是0-1归一化，这里设为1.0或根据需要调整
PIXEL_SCALE = 1.0 

# 为碳通量数据定义一个颜色映射 (从低到高: 蓝 -> 绿 -> 黄 -> 红)
COLOR_MAP = np.array([
    [0, 0, 139, 255],    # DarkBlue
    [0, 0, 255, 255],    # Blue
    [0, 255, 255, 255],  # Cyan
    [0, 255, 0, 255],    # Green
    [255, 255, 0, 255],  # Yellow
    [255, 165, 0, 255],  # Orange
    [255, 0, 0, 255],    # Red
    [139, 0, 0, 255]     # DarkRed
]) / 255.0

# 定义数据值的边界，用于颜色映射
# 假设数据在0-1之间，我们将其分成8个等级
BOUNDS = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1] 

# 定义评估指标的阈值 (同样在0-1范围内)
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

# HMF 颜色 (可以复用原来的)
HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def gray2color(image, **kwargs):
    """
    将单通道的 POC Flux 数据（灰度图）转换为彩色图。
    """
    cmap = colors.ListedColormap(COLOR_MAP)
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image


# ===============================================================
# 3. 可视化裁剪区域的测试函数 (适配服务器)
# ===============================================================
def test_and_visualize_crop(data_path, crop_config, output_filepath, variable='poc_flux'):
    """
    可视化数据集的裁剪区域，并将结果保存为图片文件。

    参数:
    - data_path (str): 数据集根目录。
    - crop_config (dict): 要测试的裁剪配置。
    - output_filepath (str): 输出图片的文件路径 (例如: './crop_visualization.png')。
    - variable (str): 要读取的变量名。
    """
    print("--- 开始可视化裁剪区域测试 (服务器模式) ---")

    try:
        temp_dataset = PocFluxDataset(mode='test', data_path=data_path, seq_len=1, img_size=128, train_ratio=0.9, valid_ratio=0.05)
        all_files = temp_dataset._find_and_sort_files()
        if not all_files:
            raise FileNotFoundError("在指定路径下找不到任何HDF文件。")
    except Exception as e:
        print(f"错误: 无法初始化数据集或查找文件。请检查路径 '{data_path}'。错误: {e}")
        return

    global_map_path = all_files[0]
    print(f"使用文件 '{os.path.basename(global_map_path)}' 作为全球背景图。")
    try:
        hdf = SD(global_map_path, SDC.READ)
        sds = hdf.select(variable)
        global_data = sds.get().astype(np.float32)
        global_data[global_data == -9999.] = np.nan
        hdf.end()
    except Exception as e:
        print(f"错误: 读取HDF文件 '{global_map_path}' 失败。错误: {e}")
        return

    print(f"使用 crop_config 加载数据: {crop_config}")
    dataset = PocFluxDataset(mode='test', data_path=data_path, seq_len=1,
                             img_size=crop_config['height'],
                             variable=variable,
                             crop_config=crop_config,
                             train_ratio=0.9, valid_ratio=0.05)

    if len(dataset) == 0:
        print("错误：数据集未能加载任何样本，无法生成图像。")
        return

    cropped_tensor = dataset[0]
    cropped_image = cropped_tensor.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 子图1: 全球地图和裁剪框
    ax1 = axes[0]
    im1 = ax1.imshow(global_data, cmap='plasma', vmin=np.nanmin(global_data), vmax=np.nanpercentile(global_data, 99))
    ax1.set_title(f'Global Map with Crop Area\n(Full Shape: {global_data.shape})')
    ax1.set_xlabel('Longitude Index')
    ax1.set_ylabel('Latitude Index')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=variable)
    rect = patches.Rectangle((crop_config['left'], crop_config['top']), crop_config['width'], crop_config['height'],
                             linewidth=2.5, edgecolor='r', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    padding = max(crop_config['width'], crop_config['height']) * 1.5
    ax1.set_xlim(max(0, crop_config['left']-padding), min(global_data.shape[1], crop_config['left']+crop_config['width']+padding))
    ax1.set_ylim(min(global_data.shape[0], crop_config['top']+crop_config['height']+padding), max(0, crop_config['top']-padding))

    # 子图2: 数据集加载的裁剪/处理后图像
    ax2 = axes[1]
    im2 = ax2.imshow(cropped_image, cmap='plasma', vmin=0, vmax=1)
    ax2.set_title(f'Actual Dataset Output (Cropped & Processed)\n(Final Shape: {cropped_image.shape})')
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('Pixel')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Normalized Value')

    plt.suptitle(f'Verification of Crop Config: top={crop_config["top"]}, left={crop_config["left"]}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # !! 核心改动: 保存文件而不是显示
    try:
        plt.savefig(output_filepath, bbox_inches='tight', dpi=150)
        print(f"\n成功! 可视化图像已保存到: {os.path.abspath(output_filepath)}")
    except Exception as e:
        print(f"\n错误: 保存图像失败。请检查路径 '{output_filepath}' 是否有效且有写入权限。错误: {e}")
    
    plt.close(fig) # 关闭图表，释放内存
    print("--- 测试完成 ---")


# ===============================================================
# 4. 主程序入口 (适配服务器)
# ===============================================================
if __name__ == '__main__':
    # --- 配置参数 ---
    from matplotlib import colors
    # 确保在无GUI的服务器上正常运行
    import matplotlib
    matplotlib.use('Agg') # !! 关键: 切换到非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # !! 重要: 请确保此路径指向您存放数据的根目录
    DATA_ROOT_PATH = '/data/fcj/ocean/code/data/'

    # 定义一个你想测试的裁剪区域
    CROP_CONFIG_TO_TEST = {
        'top': 600,
        'left': 3600,
        'height': 128,
        'width': 128
    }
    
    # !! 新增: 指定输出图片的文件名和路径
    #    可以指定绝对路径 (如 '/home/user/output.png')
    #    或相对路径 (如 'visualization.png')
    OUTPUT_IMAGE_PATH = 'crop_visualization_test.png'

    # --- 运行测试 ---
    if not os.path.isdir(DATA_ROOT_PATH):
        print(f"错误：指定的数据路径不是一个有效的目录: '{DATA_ROOT_PATH}'")
        print("请在脚本中修改 'DATA_ROOT_PATH' 变量为您的正确路径。")
    else:
        test_and_visualize_crop(
            data_path=DATA_ROOT_PATH,
            crop_config=CROP_CONFIG_TO_TEST,
            output_filepath=OUTPUT_IMAGE_PATH
        )