import torch 
import torch.nn as nn
from torch.optim import (Adam, Optimizer)
from warnings import warn
from ..utils.general_utils import build_rotation
from ..types import *


from collections import namedtuple
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
            {"params": [self.xyz], "lr": opt_cfg.xyz_lr_rate, "name": "xyz"},
            {"params": [self.scales], "lr": opt_cfg.scales_lr_rate, "name": "scales"},
            {"params": [self.quats], "lr": opt_cfg.quats_lr_rate, "name": "quats"},
            {"params": [self.opacities], "lr": opt_cfg.opacities_lr_rate, "name": "opacities"},
            {"params": [self.features_dc], "lr": opt_cfg.features_dc_lr_rate, "name": "features_dc"},
            {"params": [self.features_rest], "lr": opt_cfg.features_rest_lr_rate, "name": "features_rest"}
        ])
        if opt_cfg.optimize_cams:
            positions = nn.Parameter(torch.zeros(self.viewpoints_N, 3))
            quats = nn.Parameter(torch.zeros(self.viewpoints_N, 4))
            self.cams_optimizer = Adam([
                {"params": [positions], "name": "positions", "lr": opt_cfg.cams_t_lr_rate},
                {"params": [quats], "name": "orientation", "lr": opt_cfg.cams_q_lr_rate}
            ])
    

class DensificationStrategyDefault:
    def __init__(
        self,
        cams_extent: float,
        max_screen_size: float,
        densify_from_step: Optional[int]=None,
        densify_until_step: Optional[int]=None,
        densification_interval: Optional[int]=None,
        opacity_threshold: Optional[float]=0.2,
        gradient_threshold: Optional[float]=0.1,
        split_parts: Optional[int]=2,
        percent_dense: Optional[float]=0.1,
        device: Optional[str]="cuda"
    ) -> None:
        
        # attributes to manage densification procedure
        self._gs_models: Dict[str, GaussianModel] = {}
        self._gradient_accums: Dict[str, TensorType["Npoints", "XYZ"]] = {}
        self._denoms: Dict[str, float] = {}

        self.cams_extent = cams_extent
        self.max_screen_size = max_screen_size
        self.opacity_threshold = opacity_threshold
        self.gradient_threshold = gradient_threshold
        self.split_parts = split_parts
        self.percent_dense = percent_dense

        self._steps_schedule: Dict[str, int] = {}
        self.dens_start = densify_from_step
        self.dens_end = densify_until_step
        self.dens_interval = densification_interval
        self.device = device


    def add_gaussian_space(self, gs: GaussianModel) -> None:        
        self._gs_models.update({gs.name: gs})
        self._gradient_accums.update({gs.name: torch.zeros(gs.points_N, 1).to(self.device)})
        self._denoms.update({gs.name: torch.zeros(gs.points_N, 1).to(self.device)})
        self._steps_schedule.update({gs.name: 0})
        
    def _update_optimizer(
        self, 
        gs_model_name: str,
        input_pkg: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mode: Optional[str]="prune"
    ) -> Tuple[Dict[str, torch.Tensor], Optimizer]:
        
        model = self._gs_models.get(gs_model_name, None)
        assert (model is not None), \
        (f"couldn't find any model with name: {gs_model_name}")
        assert (hasattr(model, "optimizer")), \
            (f"{gs_model_name} model doesnt have an optimizer, try to use model.training_setup(opt_config)")
        optimizer = model.optimizer
        
        optimizable_tensors = {}
        for group in optimizer.param_groups:

            if mode in ["add", "cat", "stack"]:
                assert (input_pkg is not None \
                            and (group["name"] in input_pkg) \
                            and (input_pkg[group["name"]] is not None)), \
                    (f"""to delete gaussian from optimizer you must provide correct input_pkg. 
                    Current value for this arg: {input_pkg} """)
                tensor2add = input_pkg[group["name"]]
                print(tensor2add.is_cuda, group["params"][0].is_cuda)
                new_params = torch.cat([group["params"][0], tensor2add], dim=0)

            elif mode in ["delete", "purge", "prune"]:
                new_params = group['params'][0][input_pkg]
                
            else:
                raise ValueError(f"unknow mode: {mode}")

            stored_state = optimizer.state.get(group["params"][0], None)
            optimizable_tensors.update({group["name"]: new_params})
            if stored_state is not None:
                if mode in ["delete", "purge", "prune"]:
                    stored_state["exp_avg"] = stored_state["exp_avg"][input_pkg]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][input_pkg]
                elif mode in ["add", "cat", "stack"]:
                    stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], tensor2add], dim=0)
                    stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], tensor2add], dim=0)
                    
                del optimizer.state[group["params"][0]]
                optimizer.state.update({group["params"][0]: stored_state})
        return (optimizable_tensors, optimizer)

        
    def _update_optimizers(self, update_pkg: Dict[str, Any], name: str, mode: str) -> None:
            (new_params, new_optimizer) = self._update_optimizer(name, input_pkg=update_pkg, mode=mode)
            
            # updating spaces parameters
            self._gs_models[name].optimizer = new_optimizer
            self._gs_models[name].xyz = nn.Parameter(new_params["xyz"])
            self._gs_models[name].scales = nn.Parameter(new_params["scales"])
            self._gs_models[name].quats = nn.Parameter(new_params["quats"])
            self._gs_models[name].opacities = nn.Parameter(new_params["opacities"])
            self._gs_models[name].features_dc = nn.Parameter(new_params["features_dc"])
            self._gs_models[name].features_rest = nn.Parameter(new_params["features_rest"])

            # updating densifiaction managing properties
            self._gradient_accums[name] = torch.zeros(new_params["xyz"].size(0), 1).to(self.device)
            self._denoms[name] = torch.zeros(new_params["xyz"].size(0), 1).to(self.device)
        
    def _check_step(self, name: str) -> bool:
        step = self._steps_schedule[name]
        if (step >= self.dens_start) \
            and (step <= self.dens_end) \
            and (step % self.dens_interval == 0):
            return True
        return False
    
    def _densify_and_clone(self, name: str, grads: torch.Tensor):
        gs = self._gs_models[name]
        mask_grad = (torch.linalg.norm(grads, dim=-1) >= self.gradient_threshold)
        mask_size = (gs.scales.max(dim=-1).values <= self.percent_dense * self.cams_extent)
        mask = (mask_grad | mask_size)
        self._update_optimizers({
            "xyz": gs.xyz[mask],
            "scales": gs.scales[mask],
            "quats": gs.quats[mask],
            "opacities": gs.opacities[mask],
            "features_dc": gs.features_dc[mask],
            "features_rest": gs.features_rest[mask]
        }, name, "add")
    
    def _densify_and_split(self, name: str, grads: torch.Tensor) -> None:
        gs = self._gs_models[name]
        mask_grad = (torch.linalg.norm(grads, dim=-1) >= self.gradient_threshold)
        mask_size = (gs.scales.max(dim=-1).values >= self.percent_dense * self.cams_extent)
        mask = (mask_grad | mask_size)

        scales = gs.scales[mask].repeat(self.split_parts, 1)
        sampled_trans = torch.normal(mean=torch.zeros_like(scales), std=scales).to(self.device)
        Rs = build_rotation(gs.quats[mask]).repeat(self.split_parts, 1, 1)
        print(sampled_trans.size(), Rs.size())
        xyz = torch.bmm(Rs, (sampled_trans + gs.xyz[mask].repeat(self.split_parts, 1))[..., None]).squeeze()
        del sampled_trans, Rs

        self._update_optimizers({
            "xyz": xyz,
            "scales": scales / (0.8 * self.split_parts),
            "quats": gs.quats[mask].repeat(self.split_parts, 1),
            "opacities": gs.opacities[mask].repeat(self.split_parts, 1),
            "features_dc": gs.features_dc[mask].repeat(self.split_parts, 1, 1),
            "features_rest": gs.features_rest[mask].repeat(self.split_parts, 1, 1)
        }, name, "add")

    def update_state(
        self, 
        states: Dict[str, TensorType["Npoints", "XYZ"]],
        update_filters: Optional[Dict[str, TensorType["Npoints"]]]=None
    ) -> None:
        for (k, grads) in states.items():
            if grads.isnan().any():
                warn(f" [NaN VALUES IN GRADIENTS: {k}]")
                grads[grads.isnan()] = 0.0
                states.update({k: grads})

            if self._check_step(k):

                print(f"POINTS BEFORRE DENSIFICATION: {self._gs_models[k].xyz.size()}")
                self._densify_and_split(k, grads)
                self._densify_and_clone(k, grads)

                mask_op = (self._gs_models[k].opacities < self.opacity_threshold)
                mask_size = (self._gs_models[k].scales.max(dim=-1).values > self.max_screen_size)
                mask_ext = (self._gs_models[k].scales.max(dim=-1).values > 0.1 * self.cams_extent)
                mask = (mask_op | mask_ext | mask_size)
                self._update_optimizers(mask, k, "prune")
                torch.cuda.empty_cache()
                print(f"POINTS AFTER DENSIFICATION: {self._gs_models[k].xyz.size()}")

            # update steps and collect gradient accumulation
            assert (k in update_filters), \
            (f"unknown filter name: {k}")
            filter = update_filters[k]
            if filter is None:
                filter = torch.ones_like(grads).to(torch.bool)

            grads_norm = torch.linalg.norm(states[k], dim=-1, keepdim=True)
            self._gradient_accums[k][filter] += grads_norm[filter]
            self._denoms[k][filter] += 1
            self._steps_schedule[k] += 1




if __name__ == "__main__":

    opt_config = Optconfig(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, False, 0.1, 0.1)
    model = GaussianModel(
        name="test",
        viewpoints_N=1,
        points_N=100,
        xyz=torch.rand(100, 3),
        features_dc=torch.rand(100, 1, 3),
        opacities=torch.rand(100, 1),
        quats=torch.rand(100, 4),
        scales=torch.rand(100, 3)
    ).to("cuda")
    model.setup_optimizer(opt_config)
    print(type(model), model.xyz.is_cuda)
    stratagy = DensificationStrategyDefault(
        cams_extent=10,
        max_screen_size=224,
        densification_interval=2,
        densify_from_step=0,
        densify_until_step=5,
        device="cuda"
    )
    stratagy.add_gaussian_space(model)

    for _ in range(10):
        model.xyz.grad = torch.rand(100, 3).to("cuda")
        states = {model.name: model.xyz.grad}
        stratagy.update_state(states)
