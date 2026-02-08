import torch 
import torch.nn as nn
import os
import nibabel as nib
import numpy as np
import matplotlib.cm as cm
from torch.optim import (Adam, Optimizer)
from warnings import warn
from collections import namedtuple
from ..utils.general_utils import build_rotation, build_scaling_rotation
from ..types import *
from plyfile import (PlyData, PlyElement)



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
class Sensor:
    Transform: Optional[torch.Tensor]=None
    theta_max: Optional[List[float]]=None
    phi_max: Optional[List[float]]=None
    rad_max: Optional[List[float]]=None
    depths: Optional[List[float]]=None # depths of each sensor along direction

@dataclass
class GaussianModel:
    name: str
    viewpoints_N: Optional[int]=None
    # gs attributes 
    points_N: Optional[int]=None
    xyz: Optional[TensorType["Npoints", "XYZ"]]=None
    features_dc: Optional[TensorType["Npoints", "FEATURES", "RGB"]]=None
    opacities: Optional[TensorType["Npoints", "Value"]]=None
    scales: Optional[TensorType["Npoints", "XYZ"]]=None
    quats: Optional[TensorType["Npoints", "XYZW"]]=None
    features_rest: Optional[TensorType["Npoints", "FEATURES", "RGB"]]=None
    max_sh_degree: Optional[int]=2
    sensor: Optional[Sensor]=None # sensor attributes
    

    
    # def __post_init__(self) -> None:
    #     if self.features_rest is not None:
    #         self.features = torch.zeros(self.points_N, (self.features_rest.size(1) + 1), 3)
    #         self.features[:, 1:, :] = self.features_rest
    #     else:
    #         self.features = torch.zeros(self.points_N, (self.max_sh_degree + 1) ** 2, 3)
    #         self.features_rest = self.features[:, 1:, :]
    #     self.features[:, :1, :] = self.features_dc
    
    def restore(self, xyz: torch.Tensor, 
                scales: torch.Tensor,
                quats: torch.Tensor,
                opacities: torch.Tensor,
                features_dc: torch.Tensor,
                features_rest: torch.Tensor) -> None:
        self.xyz = (xyz if xyz.require_grad else nn.Parameter(xyz).requires_grad_(True))
        self.scales = (scales if scales.require_grad else nn.Parameter(scales).requires_grad_(True))
        self.quats = (quats if quats.require_grad else nn.Parameter(quats).requires_grad_(True))
        self.opacities = (opacities if opacities.require_grad else nn.Parameter().requires_grad_(True))
        self.features_dc = (features_dc if features_dc.require_grad else nn.Parameter().requires_grad_(True))
        self.features_rest = (features_rest if features_rest.require_grad else nn.Parameter().requires_grad_(True))
        

    @property
    def covariance(self):
        M = build_scaling_rotation(self.get_scales, self.quats)
        Sigma = M @ M.transpose(-1, -2)
        return Sigma
    @property
    def get_scales(self):
        print(self.scales.min(), self.scales.mean(), self.scales.max())
        scales = torch.exp(self.scales)
        print(scales.min(), scales.mean(), scales.max())
        return scales
    
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
    
    def load_nifti(self, source: str, 
                    pts_scale: Optional[float]=1.0,
                    base_volume_size: Tuple[int, int, int]=(112, 112, 112),
                    max_opacity_trashold: Optional[float]=0.65,
                    min_opacity_trashold: Optional[float]=0.32,
                    dst_coeff: Optional[float]=0.0,
                    base_transform: Optional[torch.Tensor]=None) -> None:

        assert (os.path.exists(source)), \
        (f"couldn't find any fiel at location: {source}")
        
        voxel_data = nib.load(source).get_fdata()
        # voxel_data = voxel_data[..., np.random.randint(0, voxel_data.shape[-1])]
        print(voxel_data.shape)
        assert (voxel_data.ndim == 3), \
        (f"""your nifti archive seems to be in FRMI format. 
        Only simple MRI files with voxel shape: [W, H, D]
        are exeptable for basic mri Gaussian Splatting map generation!!!""")

        voxel_data = nn.functional.interpolate(
            torch.from_numpy(voxel_data)[None, None],
            base_volume_size
        )
        max_n = max(voxel_data.shape)
        (x, y, z) = torch.meshgrid(
            torch.arange(max_n),
            torch.arange(max_n), 
            torch.arange(max_n)
        )
        points_xyz = torch.flatten(
            torch.stack([x, y, z], dim=-1), 
            end_dim=-2
        )
        points_xyz = (2 * (points_xyz / torch.max(points_xyz))) - 1
        if base_transform is not None:
            points_xyz = (base_transform @ points_xyz.T).T
        opacities = torch.flatten(voxel_data)[:, None]
        opacities = (opacities / torch.max(opacities))

        max_opacity_trashold = torch.tensor(max_opacity_trashold).to(opacities.dtype)
        min_opacity_trashold = torch.tensor(min_opacity_trashold).to(opacities.dtype)
        mask = torch.where(
            (opacities > torch.quantile(opacities, min_opacity_trashold)) \
                | (opacities < torch.quantile(opacities, max_opacity_trashold)),
            True, False
        ).squeeze()
        self.points_N = torch.sum(mask)

        opacities = opacities[mask]
        points_xyz = points_xyz[mask]
        opacities = opacities - dst_coeff * torch.linalg.norm(points_xyz, dim=-1)[:, None]
        points_xyz *= pts_scale
        
        # -------
        # features_dc = torch.from_numpy(cm.turbo(opacities.numpy())).float()
        # features_dc = torch.mul(features_dc[..., :3], features_dc[..., 3, None])
        
        # features_dc = (features_dc / torch.max(features_dc))
        # features_dc = features_dc[..., :3].transpose(-1, -2)

        features_dc = (torch.ones(self.points_N, 3) * opacities)[..., None]
        # features_dc = (features_dc / torch.max(features_dc))
        # print(features_dc.size())
        # -------

        features = torch.zeros(self.points_N, 3, (self.max_sh_degree + 1) ** 2)
        features[..., :1] = features_dc
        scales = 0.5 * torch.ones(self.points_N, 3)
        quats = torch.ones(self.points_N, 4)
        
        self.xyz = nn.Parameter(points_xyz.requires_grad_(True))
        self.features_dc = nn.Parameter(features[..., :1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[..., 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scales = nn.Parameter(scales.requires_grad_(True))
        self.quats = nn.Parameter(quats.requires_grad_(True))
        self.opacities = nn.Parameter(opacities.requires_grad_(True))

    def save_ply(self, dest: Optional[str]=None) -> None:

        if dest is None:
            dest = "g_splats.ply"
            warn("""!!!Dest arg was None file will bed saved in project current location. 
                 IF you want to change this behaviour just pass location that you wan't as
                 dest: str argument!!!""")
        
        xyz = self.xyz.detach().cpu().numpy()
        nxyz = np.zeros_like(xyz)
        scales = self.scales.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy()
        features_dc = torch.flatten(self.features_dc, start_dim=-2).detach().cpu().numpy()
        features_rest = torch.flatten(self.features_rest, start_dim=-2).detach().cpu().numpy()

        attributes = ["x", "y", "z", 
                      "nx", "ny", "nz", 
                      "opacities"]
        features_dc_attrs = [f"f_dc_{idx}" for idx in range(features_dc.shape[1])]
        features_rest_attrs = [f"f_rest_{idx}" for idx in range(features_rest.shape[1])]
        scales_attrs = [f"scale_{idx}" for idx in range(scales.shape[1])]
        rots_attrs = [f"rot_{idx}" for idx in range(quats.shape[1])]
        attributes = (attributes 
                      + features_dc_attrs 
                      + features_rest_attrs 
                      + scales_attrs 
                      + rots_attrs)
        attributes = list(zip(attributes, ["f4" for _ in range(len(attributes))]))
        print(attributes)
        elements = np.empty(self.points_N, dtype=attributes)
        elements[:] = list(map(tuple, np.concatenate([xyz, nxyz, opacities,
                                      features_dc, features_rest,
                                      scales, quats], axis=1)))
        elements = PlyElement.describe(elements, "vertex")
        PlyData([elements]).write(dest)
        print(f"Ply file was writen to the location: {dest}")
    
    def load_ply(self, source: str) -> None:

        assert (os.path.exists(source)
                and os.path.isfile(source)), \
                (f"file is not exists or its not a dfile: {source}")
        assert ("ply" in os.path.basename(source)), ("can read only .ply file")

        data = PlyData.read(source)["vertex"]
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
        scales = np.stack([data[f"scale_{i}"] for i in range(3)], axis=-1)
        quats = np.stack([data[f"rot_{i}"] for i in range(4)], axis=-1)
        opacities = data["opacities"][:, np.newaxis]
        featuers_dc = np.stack([data[f"f_dc_{i}"] for i in range(3)])[..., np.newaxis]
        print(3 * ((self.max_sh_degree + 1) ** 2))
        features_rest = np.stack([
            data[f"f_rest_{i}"] 
            for i in range(3 * (((self.max_sh_degree + 1) ** 2) - 1))
        ], axis=-1).reshape(-1, 3, ((self.max_sh_degree + 1) ** 2 - 1))
        
        self.xyz = nn.Parameter(torch.from_numpy(xyz).requires_grad_(True))
        self.scales = nn.Parameter(torch.from_numpy(scales).requires_grad_(True))
        self.quats = nn.Parameter(torch.from_numpy(quats).requires_grad_(True))
        self.opacities = nn.Parameter(torch.from_numpy(opacities).requires_grad_(True))
        self.features_dc = nn.Parameter(torch.from_numpy(featuers_dc).requires_grad_(True))
        self.features_rest = nn.Parameter(torch.from_numpy(features_rest).requires_grad_(True))
        
            
    
if __name__ == "__main__":

    gs = GaussianModel("test_brain")
    gs.load_nifti("/home/ram/Downloads/sub-02_T1w.nii")
    gs.save_ply("first_brain_test.ply")

    path = "first_brain_test.ply"
    gs = GaussianModel("test")
    gs.load_ply(path)
    
    print(gs.xyz.size(), gs.xyz.mean())
    print(gs.features_dc.size(), gs.features_rest.size())
    # print(data["vertex"].data.shape)
    # array = data["vertex"].data
    # print(array["x"].shape, array["f_dc_0"].shape)





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
