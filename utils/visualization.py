import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from matplotlib import colors
from datetime import datetime, timedelta # 新增: 用于处理日期
import pandas as pd # 新增: 用于保存CSV文件

# 假设 FCO2 的 colormap 定义在 dataloader 文件中，这里为了独立性重新定义
# 在项目内可从 `.dataset_fco2` 导入 `DIVERGING_CMAP`
DIVERGING_CMAP = 'RdBu_r'


# ===============================================================
# 模块 0: 核心库导入 (已更新)
# ===============================================================
# 新增: 从脚本1移植的、地理掩码生成所需的核心库
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep


# ===============================================================
# 模块 1: 地理掩码生成工具 (从脚本1移植)
# ===============================================================
def get_geographical_info(config):
    """根据配置计算裁剪后图像的地理经纬度范围。"""
    deg_per_lon_pixel = (config['ORIGINAL_EXTENT']['lon'][1] - config['ORIGINAL_EXTENT']['lon'][0]) / config['ORIGINAL_DIMS']['lon']
    deg_per_lat_pixel = (config['ORIGINAL_EXTENT']['lat'][1] - config['ORIGINAL_EXTENT']['lat'][0]) / config['ORIGINAL_DIMS']['lat']
    
    new_lon_min = config['ORIGINAL_EXTENT']['lon'][0] + config['CROP_PARAMS']['left'] * deg_per_lon_pixel
    new_lon_max = new_lon_min + config['CROP_PARAMS']['width'] * deg_per_lon_pixel
    new_lat_max = config['ORIGINAL_EXTENT']['lat'][1] - config['CROP_PARAMS']['top'] * deg_per_lat_pixel
    new_lat_min = new_lat_max - config['CROP_PARAMS']['height'] * deg_per_lat_pixel
    
    return [new_lon_min, new_lon_max, new_lat_min, new_lat_max]

def create_geographic_land_mask(geo_extent, height, width, mask_scale='50m'):
    """
    根据地理坐标范围和图像尺寸，生成一个精确的陆地掩码。
    True 表示陆地，False 表示海洋。
    """
    print(f"--- 正在创建地理陆地掩码 (精度={mask_scale}) ---")
    lon_min, lon_max, lat_min, lat_max = geo_extent
    
    lon_coords = np.linspace(lon_min, lon_max, width)
    lat_coords = np.linspace(lat_max, lat_min, height)
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    land_feature = cfeature.NaturalEarthFeature(
        'physical', 'land', mask_scale, edgecolor='face', facecolor='none'
    )
    land_polygons = list(land_feature.geometries())
    prepared_polygons = [prep(p) for p in land_polygons]

    mask = np.zeros(lon_grid.shape, dtype=bool)
    points_flat = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T

    for p in prepared_polygons:
        min_lon, min_lat, max_lon, max_lat = p.context.bounds
        idx = np.where(
            (points_flat[:, 0] >= min_lon) & (points_flat[:, 0] <= max_lon) &
            (points_flat[:, 1] >= min_lat) & (points_flat[:, 1] <= max_lat)
        )[0]
        
        if len(idx) == 0:
            continue
        
        res = [p.contains(Point(xy)) for xy in points_flat[idx]]
        mask.ravel()[idx] = np.logical_or(mask.ravel()[idx], res)
    
    print("--- 地理陆地掩码创建完成 ---")
    return mask


# ===============================================================
# 2. 为 FCO2 数据集定制的高级可视化函数 (已修复和增强)
# ===============================================================
def visualize_fco2_sequences(target, prediction, save_path, sample_index, norm_min, norm_max, percentile=98, start_date=None):
    """
    为 FCO2 数据序列定制的可视化函数，具有科学准确性。
    (已更新：支持批处理、基于日期的帧命名、陆地掩膜白色、分别保存预测/GT/差异图)
    """
    # --- 0. 将整个批次转换为numpy数组 ---
    batch_prediction_norm = prediction.detach().cpu().numpy()
    batch_target_norm = target.detach().cpu().numpy()

    # 若输入维度为 (B, T, 1, H, W)，去掉通道维度
    if batch_prediction_norm.ndim == 5 and batch_prediction_norm.shape[2] == 1:
        batch_prediction_norm = batch_prediction_norm.squeeze(2)
        batch_target_norm = batch_target_norm.squeeze(2)
        
    batch_size = batch_prediction_norm.shape[0]

    # <<< 已修复: 循环处理批次中的每一个样本 >>>
    for i in range(batch_size):
        current_sample_index = sample_index + i
        pred_norm = batch_prediction_norm[i]
        target_norm = batch_target_norm[i]

        # --- 1. 数据准备与反归一化 (针对单个样本) ---
        sample_name = f"fco2_sample_{current_sample_index:04d}"
        os.makedirs(save_path, exist_ok=True)
        frames_save_path = osp.join(save_path, 'frames', sample_name)
        os.makedirs(frames_save_path, exist_ok=True)
        # 为分别保存的预测/GT/差异图创建子目录
        os.makedirs(osp.join(frames_save_path, 'prediction'), exist_ok=True)
        os.makedirs(osp.join(frames_save_path, 'ground_truth'), exist_ok=True)
        os.makedirs(osp.join(frames_save_path, 'difference'), exist_ok=True)


        pred_de = pred_norm * (norm_max - norm_min) + norm_min
        target_de = target_norm * (norm_max - norm_min) + norm_min

        # --- 打印诊断信息 ---
        print(f"\n--- FCO2 可视化诊断信息 (样本 {current_sample_index}) ---")
        print(f"反归一化预测值统计: Min={pred_de.min():.3f}, Max={pred_de.max():.3f}, Mean={pred_de.mean():.3f}")
        print(f"反归一化真实值统计: Min={target_de.min():.3f}, Max={target_de.max():.3f}, Mean={target_de.mean():.3f}")
        
        diff_de = np.abs(pred_de - target_de)
        
        # --- 地理掩码生成逻辑 (保持不变) ---
        CONFIG = {
            "ORIGINAL_DIMS": {'lat': 2100, 'lon': 1300},
            "ORIGINAL_EXTENT": {'lon': [117, 130], 'lat': [21.00, 42.00]},
            "CROP_PARAMS": {'top': 800, 'left': 300, 'height': 1024, 'width': 1024}
        }
        _T, H, W = target_de.shape
        geo_extent = get_geographical_info(CONFIG)
        land_mask_2d = create_geographic_land_mask(geo_extent, H, W)
        land_mask = np.stack([land_mask_2d] * _T, axis=0)
        
        # <<< 新增：保存反归一化后的数据为CSV >>>
        # 提取所有海洋像素点（land_mask=False）的值
        ocean_pixels_gt = target_de[~land_mask]
        ocean_pixels_pred = pred_de[~land_mask]
        
        # 创建 DataFrame
        df_gt = pd.DataFrame({'truth': ocean_pixels_gt})
        df_pred = pd.DataFrame({'imputed': ocean_pixels_pred})
        
        # 定义保存路径
        gt_csv_path = osp.join(save_path, f"{sample_name}_ground_truth.csv")
        pred_csv_path = osp.join(save_path, f"{sample_name}_prediction.csv")
        
        # 保存文件
        df_gt.to_csv(gt_csv_path, index=False)
        df_pred.to_csv(pred_csv_path, index=False)
        print(f"--- 数据已保存 ---")
        print(f"Ground Truth CSV (海洋点) 已保存至: {gt_csv_path}")
        print(f"Prediction CSV (海洋点) 已保存至: {pred_csv_path}")
        print("--------------------------------------------------\n")

        # 应用陆地掩膜，将陆地设为 np.nan，后续 matplotlib 渲染 'bad' 颜色
        pred_masked = np.where(land_mask, np.nan, pred_de)
        target_masked = np.where(land_mask, np.nan, target_de)
        diff_masked = np.where(land_mask, np.nan, diff_de)


        # --- 色阶计算部分 (保持不变) ---
        # 仅使用海洋数据计算色阶
        valid_data = np.concatenate([
            target_de[~land_mask],
            pred_de[~land_mask]
        ])
        
        v_abs = np.percentile(np.abs(valid_data), percentile) if valid_data.size > 0 else max(abs(norm_min), abs(norm_max))
        
        # 可根据需要覆盖动态范围
        # vmin_data, vmax_data = -20, 20
        vmin_data, vmax_data = -v_abs, v_abs

        cmap_data = plt.get_cmap(DIVERGING_CMAP).copy()
        cmap_data.set_bad(color='white') # <<< 修改：陆地掩膜颜色为白色

        cmap_diff = plt.get_cmap('Reds').copy()
        cmap_diff.set_bad(color='white') # <<< 修改：陆地掩膜颜色为白色


        # --- 2. 生成并保存每一帧的对比图像 ---
        gif_frames = []
        num_frames = pred_de.shape[0]
        
        # <<< 新增：日期处理逻辑 >>>
        base_date = None
        if start_date:
            try:
                base_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                print(f"警告: start_date '{start_date}' 格式不正确，请使用 'YYYY-MM-DD'，将回退为帧索引。")
                start_date = None # 格式错误则禁用

        for t in range(num_frames):
            # <<< 新增: 动态生成标题和文件名 >>>
            time_info_str = f"Time Step: {t + 1}"
            frame_base_name = f'frame_{t:03d}'
            if base_date:
                current_date = base_date + timedelta(days=t)
                date_str = current_date.strftime('%Y-%m-%d')
                time_info_str = f"Date: {date_str}"
                frame_base_name = date_str

            # --- 组合图 ---
            fig_combined, axes_combined = plt.subplots(1, 3, figsize=(22, 6), dpi=100)
            
            title_combined = (f'FCO2 Sample: {sample_name} - {time_info_str}\n'
                              f'Shared Symmetric Color Scale (±{v_abs:.2f}, clipped at {percentile}th percentile)')
            fig_combined.suptitle(title_combined, fontsize=16)

            im1_combined = axes_combined[0].imshow(pred_masked[t], cmap=cmap_data, vmin=vmin_data, vmax=vmax_data)
            axes_combined[0].set_title("Model Prediction", fontsize=14)
            axes_combined[0].axis("off")
            fig_combined.colorbar(im1_combined, ax=axes_combined[0], shrink=0.8, label="FCO2 Flux (mol/m^2/yr)")

            im2_combined = axes_combined[1].imshow(target_masked[t], cmap=cmap_data, vmin=vmin_data, vmax=vmax_data)
            axes_combined[1].set_title("Ground Truth", fontsize=14)
            axes_combined[1].axis("off")
            fig_combined.colorbar(im2_combined, ax=axes_combined[1], shrink=0.8, label="FCO2 Flux (mol/m^2/yr)")

            valid_diff_data = diff_de[~land_mask]
            vmax_diff = np.percentile(valid_diff_data, percentile) if valid_diff_data.size > 0 else diff_de.max()
            im3_combined = axes_combined[2].imshow(diff_masked[t], cmap=cmap_diff, vmin=0, vmax=vmax_diff)
            axes_combined[2].set_title("Absolute Difference", fontsize=14)
            axes_combined[2].axis("off")
            fig_combined.colorbar(im3_combined, ax=axes_combined[2], shrink=0.8, label="Difference")
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            
            combined_frame_filename = osp.join(frames_save_path, f'{frame_base_name}_combined.png')
            plt.savefig(combined_frame_filename, bbox_inches='tight')

            fig_combined.canvas.draw()
            image_from_plot = np.frombuffer(fig_combined.canvas.buffer_rgba(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig_combined.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(image_from_plot[:, :, :3])
            plt.close(fig_combined)

            # --- 单独保存预测图 ---
            fig_pred = plt.figure(figsize=(W/100, H/100), dpi=100) # 尺寸适应图片
            ax_pred = fig_pred.add_axes([0, 0, 1, 1]) # 铺满整个figure
            ax_pred.imshow(pred_masked[t], cmap=cmap_data, vmin=vmin_data, vmax=vmax_data)
            ax_pred.axis("off") # 移除坐标轴
            pred_filename = osp.join(frames_save_path, 'prediction', f'{frame_base_name}_pred.png')
            plt.savefig(pred_filename, bbox_inches='tight', pad_inches=0) # 无边框
            plt.close(fig_pred)

            # --- 单独保存GT图 ---
            fig_gt = plt.figure(figsize=(W/100, H/100), dpi=100)
            ax_gt = fig_gt.add_axes([0, 0, 1, 1])
            ax_gt.imshow(target_masked[t], cmap=cmap_data, vmin=vmin_data, vmax=vmax_data)
            ax_gt.axis("off")
            gt_filename = osp.join(frames_save_path, 'ground_truth', f'{frame_base_name}_gt.png')
            plt.savefig(gt_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig_gt)

            # --- 单独保存Difference图 ---
            fig_diff = plt.figure(figsize=(W/100, H/100), dpi=100)
            ax_diff = fig_diff.add_axes([0, 0, 1, 1])
            ax_diff.imshow(diff_masked[t], cmap=cmap_diff, vmin=0, vmax=vmax_diff)
            ax_diff.axis("off")
            diff_filename = osp.join(frames_save_path, 'difference', f'{frame_base_name}_diff.png')
            plt.savefig(diff_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig_diff)


        # --- 3. 保存GIF ---
        gif_path = osp.join(save_path, f"{sample_name}_comparison.gif")
        imageio.mimsave(gif_path, gif_frames, fps=2, loop=0)
        print(f"✅ FCO2 可视化结果已保存 '{sample_name}' at: {save_path}")


# ===============================================================
# 3. POC Flux 数据集的可视化函数 (已修复和增强)
# ===============================================================
def visualize_poc_sequences(target, prediction, save_path, sample_index, norm_min, norm_max, percentile=98, start_date=None):
    """
    (已更新为可处理整个批次，并支持基于日期的帧命名)
    """
    # --- 0. 批量数据准备 ---
    batch_prediction_norm = prediction.detach().cpu().numpy().squeeze(2)
    batch_target_norm = target.detach().cpu().numpy().squeeze(2)
    batch_size = batch_prediction_norm.shape[0]

    # <<< 已修复: 循环处理批次中的每一个样本 >>>
    for i in range(batch_size):
        current_sample_index = sample_index + i
        pred_norm = batch_prediction_norm[i]
        target_norm = batch_target_norm[i]

        # --- 1. 单个样本数据准备 ---
        sample_name = f"poc_sample_{current_sample_index:04d}"
        os.makedirs(save_path, exist_ok=True)
        frames_save_path = osp.join(save_path, 'frames', sample_name)
        os.makedirs(frames_save_path, exist_ok=True)

        pred_de = pred_norm * (norm_max - norm_min) + norm_min
        target_de = target_norm * (norm_max - norm_min) + norm_min
        diff_de = np.abs(pred_de - target_de)
        
        # --- 地理掩码生成逻辑 (保持不变) ---
        CONFIG = {
            "ORIGINAL_DIMS": {'lat': 2100, 'lon': 1300},
            "ORIGINAL_EXTENT": {'lon': [117, 130], 'lat': [21.00, 42.00]},
            "CROP_PARAMS": {'top': 800, 'left': 300, 'height': 1024, 'width': 1024}
        }
        _T, H, W = target_de.shape
        geo_extent = get_geographical_info(CONFIG)
        land_mask_2d = create_geographic_land_mask(geo_extent, H, W)
        land_mask = np.stack([land_mask_2d] * _T, axis=0)

        # <<< 新增: 保存反归一化后的数据为CSV文件 >>>
        # 提取所有海洋像素点 (land_mask为False的区域) 的值
        ocean_pixels_gt = target_de[~land_mask]
        ocean_pixels_pred = pred_de[~land_mask]
        
        # 创建DataFrame
        df_gt = pd.DataFrame({'truth': ocean_pixels_gt})
        df_pred = pd.DataFrame({'imputed': ocean_pixels_pred})
        
        # 定义保存路径
        gt_csv_path = osp.join(save_path, f"{sample_name}_ground_truth.csv")
        pred_csv_path = osp.join(save_path, f"{sample_name}_prediction.csv")
        
        # 保存文件
        df_gt.to_csv(gt_csv_path, index=False)
        df_pred.to_csv(pred_csv_path, index=False)
        print(f"--- 数据已保存 (样本 {current_sample_index}) ---")
        print(f"Ground Truth CSV (海洋点) 已保存至: {gt_csv_path}")
        print(f"Prediction CSV (海洋点) 已保存至: {pred_csv_path}")

        pred_masked = np.ma.masked_where(land_mask, pred_de)
        target_masked = np.ma.masked_where(land_mask, target_de)
        diff_masked = np.ma.masked_where(land_mask, diff_de)

        # --- 色阶和绘图部分 (保持不变) ---
        vmin = norm_min
        valid_target_data = target_de[~land_mask]
        valid_pred_data = pred_de[~land_mask]

        vmax = norm_max
        if valid_target_data.size > 0 or valid_pred_data.size > 0:
            combined_valid_data = np.concatenate([valid_target_data, valid_pred_data])
            vmax = np.percentile(combined_valid_data, percentile)

        cmap_data = plt.cm.get_cmap('viridis').copy()
        cmap_data.set_bad(color='lightgray')

        cmap_diff = plt.cm.get_cmap('Reds').copy()
        cmap_diff.set_bad(color='lightgray')

        # --- 2. 生成并保存帧 ---
        gif_frames = []
        num_frames = pred_de.shape[0]

        # <<< 新增: 日期处理逻辑 >>>
        base_date = None
        if start_date:
            try:
                base_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                print(f"警告: start_date '{start_date}' 格式不正确, 请使用 'YYYY-MM-DD'. 回退到使用帧索引.")
                start_date = None

        for t in range(num_frames):
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
            
            # <<< 新增: 动态生成标题和文件名 >>>
            time_info_str = f"Time Step: {t + 1}"
            frame_base_name = f'frame_{t:03d}'
            if base_date:
                current_date = base_date + timedelta(days=t)
                date_str = current_date.strftime('%Y-%m-%d')
                time_info_str = f"Date: {date_str}"
                frame_base_name = date_str

            fig.suptitle(f'Sample: {sample_name} - {time_info_str} (Shared vmax clipped at {percentile}th percentile)', fontsize=16)

            im1 = axes[0].imshow(pred_masked[t], cmap=cmap_data, vmin=vmin, vmax=vmax)
            axes[0].set_title("Model Prediction", fontsize=14)
            axes[0].axis("off")
            fig.colorbar(im1, ax=axes[0], shrink=0.8, label="POC Flux")

            im2 = axes[1].imshow(target_masked[t], cmap=cmap_data, vmin=vmin, vmax=vmax)
            axes[1].set_title("Ground Truth", fontsize=14)
            axes[1].axis("off")
            fig.colorbar(im2, ax=axes[1], shrink=0.8, label="POC Flux")

            valid_diff = diff_de[~land_mask]
            vmax_diff = np.percentile(valid_diff, percentile) if valid_diff.size > 0 else diff_de.max()
            im3 = axes[2].imshow(diff_masked[t], cmap=cmap_diff, vmin=0, vmax=vmax_diff)
            axes[2].set_title("Absolute Difference", fontsize=14)
            axes[2].axis("off")
            fig.colorbar(im3, ax=axes[2], shrink=0.8, label="Difference")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            frame_filename = osp.join(frames_save_path, f'{frame_base_name}.png') # 使用新的文件名
            plt.savefig(frame_filename, bbox_inches='tight')

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(image_from_plot[:, :, :3])
            plt.close(fig)

        # --- 3. 保存GIF ---
        gif_path = osp.join(save_path, f"{sample_name}_comparison.gif")
        imageio.mimsave(gif_path, gif_frames, fps=2, loop=0)
        print(f"✅ Enhanced POC visualization saved for '{sample_name}' at: {save_path}")


# ===============================================================
# 4. 通用可视化函数 (保持不变)
# ===============================================================
def vis_res_general(pred_seq, gt_seq, save_path, sample_index,
                    gray2color, pixel_scale, thresholds, **kwargs):
    # 此函数按设计处理单个样本，无需修改
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    
    pred_seq = np.clip(pred_seq, a_min=0., a_max=1.)
    gt_seq = np.clip(gt_seq, a_min=0., a_max=1.)
    if pred_seq.ndim == 5:
        pred_seq = pred_seq.squeeze(1)
        gt_seq = gt_seq.squeeze(1)

    sample_name = f"general_sample_{sample_index:04d}"
    save_path = osp.join(save_path, sample_name)
    os.makedirs(save_path, exist_ok=True)
    
    colored_pred = np.array([gray2color(pred_seq[i]) for i in range(len(pred_seq))], dtype=np.uint8)
    colored_gt =  np.array([gray2color(gt_seq[i]) for i in range(len(gt_seq))], dtype=np.uint8)

    imageio.mimsave(osp.join(save_path, 'pred.gif'), colored_pred, fps=4, loop=0)
    imageio.mimsave(osp.join(save_path, 'targets.gif'), colored_gt, fps=4, loop=0)

    grid_pred = np.concatenate([i for i in colored_pred], axis=0)
    grid_gt = np.concatenate([i for i in colored_gt], axis=0)
    grid_concat = np.concatenate([grid_pred, grid_gt], axis=1)
    plt.imsave(osp.join(save_path, 'comparison.png'), grid_concat)
    print(f"✅ General visualization saved for '{sample_name}' at: {save_path}")


# ===============================================================
# 5. 可视化调度中心 (已更新以传递 start_date)
# ===============================================================
def visualize_and_save(dataset_name, target, prediction, save_path, sample_index, **kwargs):
    """
    根据数据集名称，智能调用相应的可视化函数。
    (已更新以处理和传递 'start_date' 参数)
    """
    dataset_name = dataset_name.lower()
    print(f"--- Dispatching visualization for dataset: '{dataset_name}' ---")

    if dataset_name in ['pocflux', 'ecsfco2']:
        norm_min = kwargs.get('norm_min')
        norm_max = kwargs.get('norm_max')
        start_date = kwargs.get('start_date', None) # <<< 新增: 获取start_date

        if norm_min is None or norm_max is None:
            raise ValueError(f"For '{dataset_name}' dataset, 'norm_min' and 'norm_max' must be provided via kwargs.")
        
        if dataset_name == 'pocflux':
            visualize_poc_sequences(
                target=target, prediction=prediction, save_path=save_path,
                sample_index=sample_index, norm_min=norm_min, norm_max=norm_max,
                start_date=start_date # <<< 新增: 传递start_date
            )
        elif dataset_name == 'ecsfco2':
            visualize_fco2_sequences(
                target=target, prediction=prediction, save_path=save_path,
                sample_index=sample_index, norm_min=norm_min, norm_max=norm_max,
                start_date=start_date # <<< 新增: 传递start_date
            )
            
    else:
        # 对于通用可视化，如果需要也可以添加日期功能，但目前保持不变
        gray2color = kwargs.get('gray2color')
        pixel_scale = kwargs.get('pixel_scale')
        thresholds = kwargs.get('thresholds')
        if not all([gray2color, pixel_scale, thresholds]):
            raise ValueError(f"For '{dataset_name}', 'gray2color', 'pixel_scale', and 'thresholds' must be provided.")

        # 通用函数通常一次只处理一个样本，所以我们循环调用它
        batch_size = prediction.shape[0]
        for i in range(batch_size):
            vis_res_general(
                pred_seq=prediction[i], gt_seq=target[i], save_path=save_path,
                sample_index=sample_index + i, gray2color=gray2color,
                pixel_scale=pixel_scale, thresholds=thresholds
            )