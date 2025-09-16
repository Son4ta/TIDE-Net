import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 关键的 diffusers 库
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# 导入我们修改过的 Dataset 类
# 确保你的环境中存在 dataset_pocflux.py 文件
from dataset_pocflux import PocFluxDataset

# 导入 scikit-image 中的评估指标
# 如果尚未安装，请运行: pip install scikit-image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# 确保在无GUI的服务器上正常运行
import matplotlib
matplotlib.use('Agg')

def mean_absolute_error(img1, img2):
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(img1 - img2))

# ===============================================================
# 0. 辅助函数：创建或加载陆-海掩码 (无变动)
# ===============================================================
def create_or_load_land_mask(config, sample_data):
    """
    根据样本数据创建或从磁盘加载一个陆-海掩码。
    海洋区域为1，陆地为0。
    """
    mask_path = os.path.join(config.output_dir, "land_sea_mask.pt")
    if os.path.exists(mask_path):
        print(f"加载已存在的掩码文件: {mask_path}")
        mask = torch.load(mask_path)
        return mask
    print("从数据创建新的陆-海掩码...")
    # 假设陆地像素值接近于0或一个非常小的值
    land_sea_mask = (sample_data[0, 0] > 1e-6).float()
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    torch.save(land_sea_mask, mask_path)
    print(f"新的掩码已创建并保存到: {mask_path}")
    return land_sea_mask


# ===============================================================
# 1. 配置参数 (无变动)
# ===============================================================
@dataclass
class TrainingConfig:
    data_root_path = '/data/fcj/ocean/code/data/' 
    variable = 'poc_flux'          # 变量名
    image_size = 256               # 图像分辨率
    crop_height = image_size       # 裁剪高度
    crop_width = image_size        # 裁剪宽度
    crop_top = 600                 # 裁剪区域的上边界
    crop_left = 3600               # 裁剪区域的左边界
    num_epochs = 50                # 训练轮次
    learning_rate = 5e-5           # 学习率
    batch_size = 2                 # 批处理大小
    train_ratio = 0.9              # 训练集比例
    dim = 64                       # UNet 基础维度
    predict_residual: bool = True  # 是否预测残差
    use_masked_loss: bool = True   # 是否使用掩码损失函数
    output_dir = "diffusion_poc_flux_output_final" # 输出目录
    model_save_path = os.path.join(output_dir, "conditional_unet_final.pth")
    visualization_save_path = os.path.join(
        output_dir, 
        f"test_visualization_dim{dim}_{image_size}px.png"
    )
    residual_norm_params: dict = field(default_factory=dict) # 存储残差归一化参数


# ===============================================================
# 2. 推理、评估与可视化函数 (核心修改)
# ===============================================================
def generate_and_visualize(config, model, scheduler, test_dataset, land_sea_mask_np):
    """
    使用训练好的模型进行推理，在归一化空间计算评估指标，
    然后反归一化并可视化结果。
    """
    if len(test_dataset) == 0:
        print("测试集为空，跳过可视化和评估。")
        return

    print("开始生成、评估和可视化测试样本...")
    model.eval()
    
    # --- 步骤 1: 数据准备 ---
    # 获取测试样本、数据范围和设备
    sample_tensor_0_1 = test_dataset[0]
    data_min_val, data_max_val = test_dataset.min_val, test_dataset.max_val
    device = model.device

    # 将数据从 [0, 1] 归一化到 [-1, 1]
    sample_tensor_minus1_1 = sample_tensor_0_1 * 2.0 - 1.0
    
    # 分离条件帧和真实目标帧
    condition_frame = sample_tensor_minus1_1[0].unsqueeze(0).to(device)
    ground_truth_frame = sample_tensor_minus1_1[1].unsqueeze(0).to(device)
    
    # --- 步骤 2: 推理过程 (生成预测) ---
    generator = torch.Generator(device=device).manual_seed(42)
    # 从一个随机噪声开始
    generated_tensor = torch.randn(condition_frame.shape, generator=generator, device=device)
    
    scheduler.set_timesteps(1000)
    for t in tqdm(scheduler.timesteps, desc="[推理] 生成预测结果"):
        with torch.no_grad():
            # 将条件帧和带噪图像拼接作为模型输入
            model_input = torch.cat([condition_frame, generated_tensor], dim=1)
            noise_pred = model(model_input, t, return_dict=False)[0]
        # 使用调度器更新图像
        generated_tensor = scheduler.step(noise_pred, t, generated_tensor, generator=generator).prev_sample

    # --- 步骤 3: 在归一化空间 [-1, 1] 中计算评估指标 ---
    print("在归一化空间 [-1, 1] 中计算评估指标...")

    # A. 准备真实图像和预测图像 (均为 [-1, 1] 范围)
    truth_norm = ground_truth_frame.squeeze().cpu().numpy()
    
    # B. 根据模式获取预测图像
    if config.predict_residual:
        # 如果模型预测的是残差，需要将其加回到条件帧上
        # 1. 将预测的归一化残差 [-1, 1] 反归一化为物理残差
        res_min = config.residual_norm_params['min']
        res_max = config.residual_norm_params['max']
        generated_residual_normalized = generated_tensor
        generated_residual_physical = (generated_residual_normalized + 1) / 2 * (res_max - res_min) + res_min
        
        # 2. 将条件帧 [-1, 1] 反归一化为物理值
        condition_frame_physical = (condition_frame + 1) / 2 * (data_max_val - data_min_val) + data_min_val
        
        # 3. 计算物理空间中的预测结果
        generated_image_physical = condition_frame_physical + generated_residual_physical
        
        # 4. **关键**: 将物理空间的预测结果重新归一化到 [-1, 1] 以便评估
        generated_norm_tensor = (generated_image_physical - data_min_val) / (data_max_val - data_min_val) * 2.0 - 1.0
        pred_norm = generated_norm_tensor.squeeze().cpu().numpy()
    else:
        # 如果模型直接预测图像，其输出已经是 [-1, 1] 范围
        pred_norm = generated_tensor.squeeze().cpu().numpy()

    # C. 应用掩码，确保只在海洋区域进行比较
    truth_norm_ocean = truth_norm * land_sea_mask_np
    pred_norm_ocean = pred_norm * land_sea_mask_np

    # D. 计算指标
    # 提取海洋区域的像素值用于计算 MSE 和 MAE
    ocean_pixels_true_norm = truth_norm_ocean[land_sea_mask_np == 1]
    ocean_pixels_pred_norm = pred_norm_ocean[land_sea_mask_np == 1]
    
    mse = mean_squared_error(ocean_pixels_true_norm, ocean_pixels_pred_norm)
    mae = mean_absolute_error(ocean_pixels_true_norm, ocean_pixels_pred_norm)
    
    # 对于PSNR和SSIM，data_range 对于 [-1, 1] 范围的数据是 2.0
    psnr = peak_signal_noise_ratio(truth_norm_ocean, pred_norm_ocean, data_range=2.0)
    ssim = structural_similarity(truth_norm_ocean, pred_norm_ocean, data_range=2.0, win_size=7)

    print("\n" + "="*45)
    print("模型性能评估 (在归一化空间 [-1, 1] 上计算):")
    print(f"  均方误差 (MSE)         : {mse:.6f}")
    print(f"  平均绝对误差 (MAE)     : {mae:.6f}")
    print(f"  峰值信噪比 (PSNR)      : {psnr:.2f} dB")
    print(f"  结构相似性 (SSIM)      : {ssim:.4f}")
    print("="*45 + "\n")

    # --- 步骤 4: 反归一化并进行可视化 ---
    print("反归一化数据并生成可视化图像...")

    # 定义一个辅助函数用于反归一化和掩码应用
    def postprocess_for_viz(tensor_minus1_1):
        tensor_physical = (tensor_minus1_1.to('cpu') + 1) / 2 * (data_max_val - data_min_val) + data_min_val
        img_np = tensor_physical.squeeze().numpy()
        return img_np * land_sea_mask_np

    # 获取用于可视化的物理值图像
    cond_img = postprocess_for_viz(condition_frame)
    truth_img = postprocess_for_viz(ground_truth_frame)
    # 对于预测图像，需要从之前计算好的归一化版本或物理值版本开始
    if config.predict_residual:
        # 使用步骤3中已计算的物理值
        gen_img = generated_image_physical.squeeze().cpu().numpy() * land_sea_mask_np
    else:
        # 从模型直接输出的 [-1, 1] tensor进行转换
        gen_img = postprocess_for_viz(generated_tensor)

    # 绘制图像
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    vmin = min(cond_img.min(), truth_img.min(), gen_img.min())
    vmax = max(cond_img.max(), truth_img.max(), gen_img.max())

    # 子图1: 条件帧
    im1 = axes[0].imshow(cond_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Condition (Frame t)\nMin: {cond_img.min():.4f}, Max: {cond_img.max():.4f}')
    axes[0].axis('off')

    # 子图2: 真实帧
    im2 = axes[1].imshow(truth_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Ground Truth (Frame t+1)\nMin: {truth_img.min():.4f}, Max: {truth_img.max():.4f}')
    axes[1].axis('off')

    # 子图3: 预测帧
    im3 = axes[2].imshow(gen_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Prediction (Generated)\nSSIM: {ssim:.4f}, MAE: {mae:.4f}')
    axes[2].axis('off')
    
    # 添加颜色条和主标题
    fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04, label=f'Physical {config.variable}')
    fig.suptitle(f'Conditional Diffusion Model Prediction (Epoch: {config.num_epochs})', fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    plt.savefig(config.visualization_save_path)
    plt.close(fig)
    print(f"可视化结果已保存到: {config.visualization_save_path}")


# ===============================================================
# 3. 训练主函数 (无变动)
# ===============================================================
def train_model():
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    if not os.path.isdir(config.data_root_path):
        print(f"错误: 数据路径 '{config.data_root_path}' 不存在。请修改 TrainingConfig 中的路径。")
        return

    print("加载数据...")
    crop_config = {'top': config.crop_top, 'left': config.crop_left, 'height': config.crop_height, 'width': config.crop_width}
    
    # 初始化训练数据集和加载器
    train_dataset = PocFluxDataset(mode='train', data_path=config.data_root_path, seq_len=2, img_size=config.image_size, variable=config.variable, crop_config=crop_config, train_ratio=config.train_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 创建或加载陆-海掩码
    sample_data_for_mask = train_dataset[0]
    land_sea_mask = create_or_load_land_mask(config, sample_data_for_mask)

    # 如果预测残差，需要预先计算残差的范围用于归一化
    if config.predict_residual:
        print("正在计算残差的归一化参数...")
        all_residuals = []
        # 使用一个临时数据加载器遍历整个训练集
        temp_loader = DataLoader(train_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)
        for batch_0_1 in tqdm(temp_loader, desc="[预处理] 计算残差范围"):
            data_min, data_max = train_dataset.min_val, train_dataset.max_val
            # 反归一化得到物理值
            batch_physical = batch_0_1 * (data_max - data_min) + data_min
            # 计算物理残差 (t+1 时刻 - t 时刻)
            residual = batch_physical[:, 1] - batch_physical[:, 0]
            # 应用掩码，只考虑海洋区域的残差
            masked_residual = residual.squeeze(1) * land_sea_mask
            # 收集所有非零（海洋）的残差值
            all_residuals.append(masked_residual[masked_residual != 0].view(-1))
        
        all_residuals = torch.cat(all_residuals)
        config.residual_norm_params['min'] = all_residuals.min().item()
        config.residual_norm_params['max'] = all_residuals.max().item()
        print(f"海洋区域残差范围计算完成: Min={config.residual_norm_params['min']:.4f}, Max={config.residual_norm_params['max']:.4f}")

    print("初始化模型...")
    # 定义 UNet 模型结构
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=2,  # 输入通道: 条件帧 + 带噪目标帧
        out_channels=1, # 输出通道: 预测的噪声 (或目标)
        layers_per_block=2,
        block_out_channels=(config.dim, config.dim, config.dim*2, config.dim*2, config.dim*4, config.dim*4),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", 
            "AttnDownBlock2D", "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", 
            "UpBlock2D", "UpBlock2D"
        ),
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 准备掩码以便在训练中广播
    land_sea_mask_bcast = land_sea_mask.to(device).unsqueeze(0).unsqueeze(0)
    
    # 初始化调度器和优化器
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    print(f"开始在 {device} 上进行训练 (残差模式: {config.predict_residual}, 掩码损失: {config.use_masked_loss})...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for step, batch_0_1 in enumerate(train_dataloader):
            batch_0_1 = batch_0_1.to(device)
            # 将数据从 [0, 1] 归一化到 [-1, 1]
            batch_minus1_1 = batch_0_1 * 2.0 - 1.0
            condition_frames = batch_minus1_1[:, 0]

            # 准备目标 `clean_targets`
            if config.predict_residual:
                # 目标是归一化后的残差
                data_min, data_max = train_dataset.min_val, train_dataset.max_val
                batch_physical = batch_0_1 * (data_max - data_min) + data_min
                target_residuals_physical = batch_physical[:, 1] - batch_physical[:, 0]
                res_min, res_max = config.residual_norm_params['min'], config.residual_norm_params['max']
                # 将物理残差归一化到 [0, 1]
                target_residuals_0_1 = (target_residuals_physical - res_min) / (res_max - res_min)
                # 再归一化到 [-1, 1]
                clean_targets = target_residuals_0_1 * 2.0 - 1.0
            else:
                # 目标是归一化后的图像
                clean_targets = batch_minus1_1[:, 1]

            # 前向扩散过程
            noise = torch.randn(clean_targets.shape).to(device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_targets.shape[0],), device=device).long()
            noisy_targets = scheduler.add_noise(clean_targets, noise, timesteps)
            
            # 模型预测
            model_input = torch.cat([condition_frames, noisy_targets], dim=1)
            noise_pred = model(model_input, timesteps, return_dict=False)[0]

            # 计算损失
            if config.use_masked_loss:
                # 只计算海洋区域的损失
                loss_pixelwise = F.mse_loss(noise_pred, noise, reduction='none')
                loss = (loss_pixelwise * land_sea_mask_bcast).sum() / land_sea_mask_bcast.sum()
            else:
                # 计算整个图像的损失
                loss = F.mse_loss(noise_pred, noise)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
        progress_bar.close()

    print("训练完成。")
    torch.save(model.state_dict(), config.model_save_path)
    print(f"模型已保存到: {config.model_save_path}")

    # --- 训练结束后进行测试和可视化 ---
    print("准备测试集和最终可视化...")
    # 使用训练集的归一化参数来初始化测试集，确保数据尺度一致
    norm_params = {'min_val': train_dataset.min_val, 'max_val': train_dataset.max_val}
    test_dataset = PocFluxDataset(
        mode='test', data_path=config.data_root_path, seq_len=2, img_size=config.image_size,
        variable=config.variable, crop_config=crop_config, train_ratio=config.train_ratio,
        precomputed_norm_params=norm_params
    )
    
    generate_and_visualize(config, model, scheduler, test_dataset, land_sea_mask.cpu().numpy())

# ===============================================================
# 4. 主程序入口
# ===============================================================
if __name__ == '__main__':
    train_model()