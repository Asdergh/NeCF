import torch 
import torch.nn as nn
from torch.optim import (Adam, Optimizer)
from warnings import warn
from collections import namedtuple
from ..utils.general_utils import build_rotation
from ..types import *


Optconfig = namedtuple("OptConfig", [
    "xyz_lr_rate",
    "scales_lr_rate",
    "quats_lr_rate",
    "opacities_lr_rate",
    "features_dc_lr_rate",
    "features_rest_lr_rate",
    "optimize_cams",
    "cams_t_lr_rate",
    "cams_q_lr_rate"
])

@dataclass
class GaussianModel:
    name: str
    viewpoints_N: int
    points_N: int
    xyz: TensorType["Npoints", "XYZ"]
    features_dc: TensorType["Npoints", "FEATURES", "RGB"]
    opacities: TensorType["Npoints", "Value"]
    scales: TensorType["Npoints", "XYZ"]
    quats: TensorType["Npoints", "XYZW"]
    features_rest: TensorType["Npoints", "FEATURES", "RGB"]=None
    max_sh_degree: Optional[int]=2
    
    def __post_init__(self) -> None:
        if self.features_rest is not None:
            self.features = torch.zeros(self.points_N, (self.features_rest.size(1) + 1), 3)
            self.features[:, 1:, :] = self.features_rest
        else:
            self.features = torch.zeros(self.points_N, (self.max_sh_degree + 1) ** 2, 3)
            self.features_rest = self.features[:, 1:, :]
        self.features[:, :1, :] = self.features_dc
    
    def to(self, arg: str) -> Self:
        attrs = ["xyz", "features_dc", 
                 "features_rest", "features",
                 "opacities", "quats",
                 "scales"]
        for attr in attrs:
            value = getattr(self, attr)
            if arg == "cuda":
                value = value.to(arg)
            elif arg == "cpu":
                value = value.detach().cpu()
            else:
                raise ValueError(f"unknown conversion type: {arg}")
            setattr(self, attr, value)
        return self
                
    
    def setup_optimizer(self, opt_cfg) -> None:
        self.optimizer = Adam([
            {"params": [nn.Parameter(self.xyz.requires_grad_(True))], "lr": opt_cfg.xyz_lr_rate, "name": "xyz"},
            {"params": [nn.Parameter(self.scales.requires_grad_(True))], "lr": opt_cfg.scales_lr_rate, "name": "scales"},
            {"params": [nn.Parameter(self.quats.requires_grad_(True))], "lr": opt_cfg.quats_lr_rate, "name": "quats"},
            {"params": [nn.Parameter(self.opacities.requires_grad_(True))], "lr": opt_cfg.opacities_lr_rate, "name": "opacities"},
            {"params": [nn.Parameter(self.features_dc.requires_grad_(True))], "lr": opt_cfg.features_dc_lr_rate, "name": "features_dc"},
            {"params": [nn.Parameter(self.features_rest.requires_grad_(True))], "lr": opt_cfg.features_rest_lr_rate, "name": "features_rest"}
        ])
        if opt_cfg.optimize_cams:
            positions = nn.Parameter(torch.zeros(self.viewpoints_N, 3).requires_grad_(True))
            quats = nn.Parameter(torch.zeros(self.viewpoints_N, 4).requires_grad_(True))
            self.cams_optimizer = Adam([
                {"params": [positions], "name": "positions", "lr": opt_cfg.cams_t_lr_rate},
                {"params": [quats], "name": "orientation", "lr": opt_cfg.cams_q_lr_rate}
            ])
    






# if __name__ == "__main__":

#     opt_config = Optconfig(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, False, 0.1, 0.1)
#     model = GaussianModel(
#         name="test",
#         viewpoints_N=1,
#         points_N=100,
#         xyz=torch.rand(100, 3),
#         features_dc=torch.rand(100, 1, 3),
#         opacities=torch.rand(100, 1),
#         quats=torch.rand(100, 4),
#         scales=torch.rand(100, 3)
#     ).to("cuda")
#     model.setup_optimizer(opt_config)
#     stratagy = DensificationStrategyDefault(
#         cams_extent=10,
#         max_screen_size=224,
#         densification_interval=100,
#         densify_from_step=0,
#         densify_until_step=1000,
#         device="cuda"
#     )
#     stratagy.add_gaussian_space(model)

    
#     import torch.nn.functional as F
#     for idx in range(1000):
#         # model.xyz.grad = torch.rand(100, 3).to("cuda")
#         t_xyz = torch.rand(model.xyz.size(0), 3).to("cuda")
#         t_scales = torch.rand(model.xyz.size(0), 3).to("cuda")
#         t_features_dc = torch.rand(model.xyz.size(0), 1, 3).to("cuda")
#         t_features_rest = torch.rand(model.xyz.size(0), 7, 3).to("cuda")
#         t_quats = torch.rand(model.xyz.size(0), 4).to("cuda")
#         t_opacities = torch.rand(model.xyz.size(0), 1).to("cuda")

#         loss_xyz = F.mse_loss(t_xyz, model.xyz)
#         loss_scales = F.mse_loss(t_scales, model.scales)
#         loss_opacities = F.mse_loss(t_opacities, model.opacities)
#         loss_quats = F.mse_loss(t_opacities, model.opacities)
#         loss_rgb = F.mse_loss(t_features_dc, model.features_dc)
#         loss_sh = F.mse_loss(t_features_rest, model.features_dc)
#         loss = loss_xyz + loss_scales + loss_opacities + loss_quats + loss_rgb + loss_rgb + loss_sh
#         loss.backward()
#         # print(model.xyz.grad.size())

#         # states = {model.name: model.xyz.grad}
#         stratagy.update_state()
