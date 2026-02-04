import torch 
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
# from gaussian_model import GaussianModel
from warnings import warn
from ..types import *



__activations__ = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax
}
get_activation = lambda act: (
    nn.Identity() if act not in __activations__ else
    __activations__[act](dim=-1) if act == "softmax" else
    __activations__[act]()
)


@dataclass
class CognetiveFieldConfig:    
    # TODO add DPM Shceduler
    samplers: Optional[str]="default"
    scheculer_type: Optional[str]="default"
    diffusion_timestampts: Optional[int]=10
    # model params
    scaling_factor: Optional[float]=2.0
    scaling_depth: Optional[int]=3
    patch_size: Optional[Tuple[int, int]]=(32, 32)
    hiden_features_size: Optional[int]=128 # features must be vividable to 4
    output_features_size: Optional[int]=312
    apply_temporal_pos_encoding: Optional[bool]=True
    apply_rope: Optional[bool]=True
    learnable_view_encoding: Optional[bool]=True
    hiden_activations: Optional[str]="relu"

def tensor_stats(x: torch.Tensor) -> None:
    print(x.min(), x.mean(), x.max())
    print(x.size())

def _spatial2sequece(x: torch.Tensor, format: str="sequence") -> torch.Tensor:
    if x.ndim == 4:
        if format == "sequence":
            x = torch.flatten(x, start_dim=1, end_dim=-2)
        elif format == "flat":
            x = x.permute(0, -1, 1, 2)
            x = torch.flatten(x, start_dim=1)
        else:
            raise ValueError(f"unknown format type: {format}")
    return x
    
def _sequence2spatial(
    x: torch.Tensor, 
    patch_size: Tuple[int, int], f: int
) -> torch.Tensor:
    if x.ndim == 3:
        x = x.view(x.size(0), *patch_size, f)
    elif x.ndim == 2:
        x = x.view(x.size(0), *patch_size, f)
        warn("""Latent tensor have form [N, Pw*Ph*C]. 
                Be sure that features stacked in right order""")
    return x


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, cfg: CognetiveFieldConfig) -> None:
        super(TemporalPositionalEncoding, self).__init__()
        self.cfg = cfg

    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[float]=1.0, 
        patch_size: Optional[Tuple[int, int]]=None
    ) -> torch.Tensor:
        patch_size = (patch_size 
                      if patch_size is not None 
                      else self.cfg.patch_size)
        d = self.cfg.hiden_features_size
        arg = lambda t, i, j: (t * torch.pi * i) / (1000**((2*j) / d))        
        pe_k = torch.stack([
            torch.cat([
                torch.Tensor([math.sin(arg(t, i, j)), math.cos(arg(t, i, j))]) 
                for j in range(patch_size[0] // 2)
            ], dim=0) for i in range(patch_size[1])
        ], dim=0)[None, ..., None]
        return (pe_k * x)
    
class ViewPoseEncoding6Dof(nn.Module):
    def __init__(self, cfg: CognetiveFieldConfig) -> None:
        super(ViewPoseEncoding6Dof, self).__init__()
        self.cfg = cfg
        if self.cfg.learnable_view_encoding:
            self._params = nn.Parameter(torch.rand(2, 4, 4))
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        
        diag = torch.eye(self.cfg.hiden_features_size // 4)
        if self.cfg.learnable_view_encoding:
            pose = (self._params[0][None] * pose) + self._params[1][None]
        pose_map = torch.kron(diag, pose)
        return pose_map


class Attention(nn.Module):
    def __init__(self, cfg: CognetiveFieldConfig) -> None:
        super(Attention, self).__init__()
        self.cfg = cfg

        self.fc = nn.Linear(self.cfg.hiden_features_size, 
                            3*self.cfg.hiden_features_size)

        if self.cfg.apply_temporal_pos_encoding:
            self.pet = TemporalPositionalEncoding(cfg)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[float]=1.0,
        patch_size: Optional[Tuple[int, int]]=None
    ) -> None:
        
        N = x.size(0)
        if self.cfg.apply_temporal_pos_encoding:
            x = _sequence2spatial(x,
                                  patch_size,
                                  self.cfg.hiden_features_size)
            x = self.pet(x, t, patch_size)
            x = _spatial2sequece(x)

        (q, k, v) = self.fc(x).view(N, 
                                    patch_size[0] \
                                    * patch_size[1] \
                                    * self.cfg.hiden_features_size, 3).unbind(dim=-1)
        print(q.size(), k.size(), v.size())
        qk = F.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.cfg.hiden_features_size))
        print(qk.size())
        qkv = qk @ v
        return qkv

class LatentFusionModel(nn.Module):
    def __init__(self, cfg: CognetiveFieldConfig) -> None:
        super(LatentFusionModel, self).__init__()
        self.cfg = cfg
        down_scales = [
            1 / (self.cfg.scaling_factor ** i) 
            for i in range(self.cfg.scaling_depth)
        ]
        up_scales = [
            self.cfg.scaling_factor ** i 
            for i in range(self.cfg.scaling_depth)
        ]
        self.scales_list = down_scales + up_scales
        self._blocks = nn.ModuleList([
            nn.ModuleDict({
                "fc": nn.Linear(self.cfg.hiden_features_size, self.cfg.hiden_features_size),
                "scale": nn.Upsample(size=(
                    int(self.cfg.patch_size[0] * scale),
                    int(self.cfg.patch_size[1] * scale)
                )),
                "att": Attention(self.cfg)
            })
            for scale in self.scales_list
        ])
        if self.cfg.apply_rope:
            self.RoPE = ViewPoseEncoding6Dof(self.cfg)
            
    def adaptive_residual(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = (x1 if x1.ndim == 4 else _sequence2spatial(
            x1,
            self.cfg.patch_size,
            self.cfg.hiden_features_size
        )).permute(0, 3, 1, 2)
        x2 = (x2 if x2.ndim == 4 else _sequence2spatial(
            x2, 
            self.cfg.patch_size,
            self.cfg.hiden_features_size
        )).permute(0, 3, 1, 2)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, x2.shape[-2:], mode="bilinear")
        return (x2 + x1).permute(0, 2, 3, 1)

    
    def forward(
        self,
        embeds: torch.Tensor,
        poses: Optional[torch.Tensor]=None,
        timestamp: Optional[float]=1.0
    ) -> torch.Tensor:
        x_last = embeds
        x = _spatial2sequece(embeds)
        for idx, block in enumerate(self._blocks): 
            
            factor = self.scales_list[idx]
            patch_size = (int(self.cfg.patch_size[0] * factor), 
                      int(self.cfg.patch_size[1] * factor))
            assert (min(patch_size) % 2 == 0 
                    and min(patch_size) / 2 > 1.0), \
            ("""patch_size must be dividable to 2 and division result 
            must be greater then one. Try so set correct patch_size argument
            into configuration class""")

            x = block["fc"](x)
            x = _sequence2spatial(x, 
                                self.cfg.patch_size, 
                                self.cfg.hiden_features_size)
            x = block["scale"](x.permute(0, 3, 1, 2))
            x = x.permute(0, 2, 3, 1)
            x = _spatial2sequece(x)
            
            rope_map = self.RoPE(poses)
            x = (rope_map @ x.transpose(-1, -2)).transpose(-1, -2)
            print("Before Attention: ", x.size())
            x = block["att"](x, timestamp, patch_size)
            x = _sequence2spatial(x,
                                  patch_size,
                                  self.cfg.hiden_features_size)
            x = self.adaptive_residual(x, x_last)
            if idx != len(self._blocks) - 1:
                x_last = x
                x = _spatial2sequece(x) 
                
        return x   
            

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    config = CognetiveFieldConfig(
        hiden_features_size=256,
        scaling_depth=2,
        patch_size=(32, 32)
    )

    latent_net = LatentFusionModel(config)
    test = torch.rand(100, *config.patch_size, config.hiden_features_size)
    poses = torch.rand(100, 4, 4)
    t = 12.3

    print(f"TOTAL PARAMETERS N: {sum([p.numel() for p in latent_net.parameters()])}")
    output = latent_net(test, poses, t)
    print(output.size())
    
    
    
    





        



