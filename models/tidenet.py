from diffusers.models.unets.unet_2d import UNet2DModel
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class FeatureRefinementModule(nn.Module):
    """
    Feature refinement module.
    Uses a UNet to refine input feature maps and correct upstream prediction bias.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=feature_dim,
            out_channels=feature_dim,
            sample_size=None,
            block_out_channels=(32, 64, 64),# (64, 128, 128)
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine the input feature map x.
        """
        residual_correction = self.unet(x, timestep=0).sample
        return x + residual_correction



class AmpTimeCell(nn.Module):
    def __init__(self, t_in, t_out, size_factor=1):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.scale = 0.02

        self.w1 = nn.Parameter((self.scale * torch.randn(2, t_in, t_out*size_factor)))
        self.b1 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out*size_factor)))
        self.w2 = nn.Parameter((self.scale * torch.randn(2, t_out*size_factor, t_out)))
        self.b2 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out)))
    
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        bias = self.tmlp(x)
        xf = torch.fft.rfft2(x, dim=[2,3], norm="ortho")
        x1_real = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[0]) - \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[1]) + \
                  self.b1[0]
        x1_imag = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[1]) + \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[0]) + \
                  self.b1[1]
        x1_real, x1_imag = F.relu(x1_real), F.relu(x1_imag)
        
        x2_real = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[0]) - \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[1]) + \
                  self.b2[0]
        x2_imag = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[1]) + \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[0]) + \
                  self.b2[1]

        x2 = torch.view_as_complex(torch.stack([x2_real, x2_imag], dim=-1))
        x = torch.fft.irfft2(x2, dim=[2,3], norm="ortho")
        x = x + bias
        return x.permute(0,4,1,2,3)


class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0,
        ):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.amptime =  AmpTimeCell(t_in, t_out)
        self.conv = nn.Sequential(nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),
                                  nn.GroupNorm(4, dim*t_out),
                                  nn.SiLU(),
                                  nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),)

    def forward(self, x):
        residual = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = self.amptime(x)
        x = x + residual

        residual = x
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x


class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2, input_shape=(128, 128), input_dim=1):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        )
        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim) for i in range(n_layers)
        ])
        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        x_ = x.permute(0,2,3,4,1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x


class AmpliNet_Unet_TypeB(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2, input_shape=(128, 128), input_dim=1):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        )
        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim) for i in range(n_layers)
        ])

        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))

        self.unet = UNet2DModel(
            sample_size=input_shape,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(hidden_dim, hidden_dim * 2),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        x_ = x.permute(0,2,3,4,1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = self.unet(x, timestep=0).sample
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x


class AmpliNet_Unet_TypeA(nn.Module):
    """
    Refactored AmpliNet.

    AmpCell and FeatureRefinementModule are unified in one nn.ModuleList
    to allow flexible, configurable insertion of refinement modules.

    Args:
        pre_seq_length (int): Input sequence length.
        aft_seq_length (int): Output sequence length (also the sequence length inside AmpCell).
        dim (int): Input/output channel dimension.
        hidden_dim (int): Hidden feature dimension.
        n_layers (int): Number of AmpCells.
        refiner_positions (list[int], optional): Indices after which to insert FeatureRefinementModule.
                                                Indexing starts from 0. For example, [0, 2] inserts after the 1st and 3rd AmpCell.
                                                Defaults to None (no refinement modules inserted).
    """
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2, input_shape=(128, 128), input_dim=1, refiner_positions=[1]):
        super().__init__()
        
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        )
        # --- Input conv ---
        self.convin = nn.Sequential(
            ResnetBlock(dim, hidden_dim), 
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # --- Unified layer container ---
        self.layers = nn.ModuleList([])
        if refiner_positions is None:
            refiner_positions = [] # default: empty list, no insertion

        # Dynamically build layer sequence
        for i in range(n_layers):
            # Add AmpCell
            # The first AmpCell uses pre_seq_length, others use aft_seq_length
            current_pre_seq_len = pre_seq_length if i == 0 else aft_seq_length
            self.layers.append(
                AmpCell(current_pre_seq_len, aft_seq_length, hidden_dim)
            )

            # Optionally insert refinement modules after this AmpCell
            if i in refiner_positions:
                self.layers.append(
                    FeatureRefinementModule(feature_dim=hidden_dim * aft_seq_length)
                )
                self.layers.append(
                    FeatureRefinementModule(feature_dim=hidden_dim * aft_seq_length)
                )

        # --- Output conv ---
        self.convout = nn.Sequential(
            ResnetBlock(hidden_dim, hidden_dim), 
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial processing
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        
        x_ = x.permute(0,2,3,4,1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')\

        # --- Core loop: process unified sequence of layers ---
        for layer in self.layers:
            if isinstance(layer, AmpCell):
                x = layer(x)
            elif isinstance(layer, FeatureRefinementModule):
                # Refinement module requires reshaping
                b, t, c, h, w = x.shape
                x_reshaped = rearrange(x, 'b t c h w -> b (t c) h w')
                x_refined = layer(x_reshaped)
                x = rearrange(x_refined, 'b (t c) h w -> b t c h w', t=t, c=c)
            # Add elif for future module types
        
        # Output processing
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x_out = self.convout(x)
        x_out = rearrange(x_out, '(b t) c h w -> b t c h w', t=self.aft_seq_length)
        
        return x_out


class PhaseNet(nn.Module):
    def __init__(self, input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim, 
                 n_layers, kernel_size, bias=1):
        super().__init__()
        h, w = input_shape
        input_shape = (h, w//2+1)
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.pha_conv0 = nn.Conv2d(2+input_dim*pre_seq_length, input_dim*aft_seq_length, 1)
        self.phase_0 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=1),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        self.phase_1 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=1),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        self.phase_2 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=3,padding_mode='circular'),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=3,padding_mode='circular'),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        
        self.pha_conv1 = nn.Conv2d(4*input_dim*aft_seq_length, input_dim*aft_seq_length, 1)
        u = torch.fft.fftfreq(h)
        v = torch.fft.rfftfreq(w)
        u, v = torch.meshgrid(u, v)
        uv = torch.stack((u,v),dim=0)
        self.register_buffer('uv', uv)

    def forward(self, x): # x:[b,t,c,h,w]
        B,T,C,H,W = x.shape
        x_fft = torch.fft.rfft2(x)
        x_amps, x_phas = torch.abs(x_fft), torch.angle(x_fft) 
        x_phas = self.pha_norm(x_phas)
        x_phas_ = rearrange(x_phas, 'b t c h w -> b (t c) h w')
        x_puv = torch.cat((x_phas_, self.uv.repeat(B,1,1,1)), dim=1)
        x_phast = self.pha_conv0(x_puv)
        x_phas0 = x_phast + self.phase_0(x_puv)
        x_phas1 = x_phast * self.phase_1(x_puv)
        x_phas2 = x_phast * self.phase_2(x_puv)
        x_phas_t = torch.cat((x_phast, x_phas0, x_phas1, x_phas2), dim=1)
        x_phas_t = self.pha_conv1(x_phas_t)
        x_phas_t = rearrange(x_phas_t, 'b (t c) h w -> b t c h w', t=self.aft_seq_length)
        x_phas_t = x_phas[:,-1:] + x_phas_t
        x_phas_t = self.pha_unnorm(x_phas_t)
        xt_fft = x_amps[:,-1:] * torch.exp(torch.tensor(1j) * x_phas_t)
        xt = torch.fft.irfft2(xt_fft)
        return xt, x_phas_t, x_amps

    def pha_norm(self, x):
        return x / torch.pi

    def pha_unnorm(self, x):
        return x * torch.pi
    
class AlphaMixer(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq_length) -> None:
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.spec_num = spec_num
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)
        self.out_mixer = nn.Sequential(ResnetBlock(3*input_dim, hidden_dim),
                                       ResnetBlock(hidden_dim, hidden_dim),
                                       nn.Conv2d(hidden_dim, input_dim, kernel_size=1))

    def forward(self, xas, xps, phas):
        xas_fft = torch.fft.rfft2(xas)
        amps = torch.abs(xas_fft)
        alpha_fft = amps * self.spec_mask * torch.exp(torch.tensor(1j) * phas)
        alpha = torch.fft.irfft2(alpha_fft)
        xap = torch.cat([xas, xps, alpha],dim=2)
        xap = rearrange(xap, 'b t c h w -> (b t) c h w')
        xt = self.out_mixer(xap)
        xt = rearrange(xt, '(b t) c h w -> b t c h w', t=self.aft_seq_length)
        return xt, alpha

# New AlphaMixer based on diffusers.UNet2DModel
class AlphaMixer_Diffusers(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq_length):
        from diffusers import UNet2DModel
        """
        AlphaMixer variant implemented with diffusers.UNet2DModel.
        """
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.spec_num = spec_num

        # Alpha generation logic remains unchanged
        spec_mask = torch.zeros(h, w // 2 + 1)
        spec_mask[..., :spec_num, :spec_num] = 1.
        spec_mask[..., -spec_num:, :spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)

        # Instantiate UNet2DModel
        # Input channels: concatenated xas, xps, alpha -> 3 * input_dim
        # Output channels: final prediction input_dim
        self.unet = UNet2DModel(
            sample_size=input_shape,
            in_channels=3 * input_dim,
            out_channels=input_dim,
            layers_per_block=2,  # number of ResNet layers per block
            block_out_channels=(hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim * 4), # U-Net channels per stage
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, xas, xps, phas):
        # 1) Alpha generation (same as original)
        xas_fft = torch.fft.rfft2(xas)
        amps = torch.abs(xas_fft)
        device = amps.device
        phas_complex = torch.exp(torch.tensor(1j, device=device) * phas)
        alpha_fft = amps * self.spec_mask * phas_complex
        alpha = torch.fft.irfft2(alpha_fft)

        # 2) Feature concatenation (same as original)
        xap = torch.cat([xas, xps, alpha], dim=2)
        xap = rearrange(xap, 'b t c h w -> (b t) c h w')

        # 3) UNet mixing and reconstruction
        # diffusers UNet does not require timestep; pass 0
        # Output is an object; use .sample
        xt = self.unet(xap, timestep=0).sample
        
        # Reshape to time sequence
        xt = rearrange(xt, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return xt, alpha


class TIDENet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, input_shape, input_dim, 
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1, 
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(TIDENet, self).__init__()

        self.amplinet = AmpliNet_Unet_TypeA(pre_seq_length, aft_seq_length, input_dim, hidden_dim, input_shape=input_shape, input_dim=input_dim)
        self.phasenet = PhaseNet(input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim, n_layers, kernel_size, bias)
        self.alphamixer = AlphaMixer_Diffusers(input_shape, spec_num, input_dim, hidden_dim, aft_seq_length)
        self.input_shape, self.input_dim = input_shape, input_dim # AlphaMixer_Diffusers
        self.hidden_dim = hidden_dim # AmpliNet_Unet_TypeA
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.criterion = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate =  self.amp_weight/self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)

    def forward(self, x, y, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = self.amplinet(x)
        xas = torch.sigmoid(xas)
        xps, x_phas_t, x_amps = self.phasenet(x)
        xt, alpha = self.alphamixer(xas, xps, x_phas_t)

        return xt, xps, xas, x_phas_t, x_amps, alpha

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        B = frames_in.shape[0]
        xt, xps, xas, x_phas_t, x_amps, alpha = self(frames_in, frames_gt, compute_loss)

        # visualize_comparison(xas, xps, alpha, xt, frames_gt)

        pred = xt
        if compute_loss:
            if self.itr < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.
            loss = 0.
            loss += self.criterion(pred, frames_gt)
            # loss += self.l1_loss(pred, frames_gt)
            frames_fft = torch.fft.rfft2(frames_gt)
            frames_pha = torch.angle(frames_fft)
            frames_abs = torch.abs(frames_fft)
            pha_loss = (1 - torch.cos(frames_pha * self.spec_mask - x_phas_t * self.spec_mask)).sum() / (self.spec_mask.sum()*B*self.aft_seq_length*self.input_dim)
            loss += self.pha_weight*pha_loss
            xas_fft = torch.fft.rfft2(xas)
            xas_abs = torch.abs(xas_fft)
            amp_loss = self.criterion(xas_abs, frames_abs)
            loss += self.amp_weight*amp_loss
            anet_loss = self.criterion(xas, frames_gt)
            loss += self.anet_weight*anet_loss
            loss = {'total_loss': loss, 'phase_loss': self.pha_weight*pha_loss,
                    'ampli_loss': self.amp_weight*amp_loss, 'anet_loss': self.anet_weight*anet_loss}
            return pred, loss
        else:
            return pred, None


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


def get_model(
    img_channels=1,
    dim = 64,
    T_in = 5, 
    T_out = 20,
    input_shape = (128,128),
    n_layers = 3,
    spec_num = 20,
    pha_weight=0.01, 
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = TIDENet(pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model



import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def visualize_comparison(xas, xps, alpha, xt, frames_gt, step=1, save_dir='debug_comparison/'):
    """
    Visualize AlphaMixer inputs, final prediction, ground truth and error map.
    Use RdBu colormap and fix range to [-1, 1].
    [This version fixes a TypeError when tensors are 3-channel.]
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tensors = {
        '1_xas': xas[0].detach().cpu(),
        '2_xps': xps[0].detach().cpu(),
        '3_alpha': alpha[0].detach().cpu(),
        '4_prediction': xt[0].detach().cpu(),
        '5_ground_truth': frames_gt[0].detach().cpu()
    }
    
    error_map = tensors['4_prediction'] - tensors['5_ground_truth']
    tensors['6_error_map'] = error_map

    fig, axes = plt.subplots(len(tensors), 1, figsize=(tensors['1_xas'].size(0) * 2, 14))
    if len(tensors) == 1:
        axes = [axes]
        
    fig.suptitle(f'Full Comparison at Step {step} (Scale: [-1, 1])', fontsize=16)
    cmap = 'RdBu_r'

    for i, (name, tensor) in enumerate(tensors.items()):
        stats_text = f"Min: {tensor.min():.4f}, Max: {tensor.max():.4f}\nMean: {tensor.mean():.4f}, Std: {tensor.std():.4f}"
        
        grid = vutils.make_grid(tensor, nrow=tensor.size(0), padding=2, normalize=False)
        
        # --- Key change ---
        # Check number of channels in grid
        if grid.shape[0] == 3:
            # If 3 channels, warn and average to single channel for colormap
            print(f"Warning: Tensor '{name}' produced a 3-channel grid. Averaging to single channel for visualization.")
            grid = grid.mean(dim=0, keepdim=True)
        # --------------------

        # Now grid is guaranteed to be single channel (1, H, W), safe to squeeze
        display_grid = grid.squeeze(0)
        
        ax = axes[i]
        im = ax.imshow(display_grid, cmap=cmap, vmin=-1, vmax=1)
        
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')
        ax.text(0, -5, stats_text, ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(save_dir, f'comparison_step_{step}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)