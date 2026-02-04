import torch 
import torch.nn as nn
from torch.optim import (Adam, Optimizer)
from warnings import warn
from .gaussian_model import GaussianModel
from ..utils.general_utils import build_rotation
from ..types import *

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
                group["params"][0] = nn.Parameter(new_params.requires_grad_(True))
                optimizer.state.update({group["params"][0]: stored_state})
            else:
                # print(group["params"][0].size(), optimizer.state.get(group["params"][0], None).size())
                group["params"][0] = nn.Parameter(new_params.requires_grad_(True))
        return (optimizable_tensors, optimizer)

        
    def _update_gs_model(self, update_pkg: Dict[str, Any], name: str, mode: str) -> None:
            (new_params, new_optimizer) = self._update_optimizer(name, input_pkg=update_pkg, mode=mode)
            points_N = new_params["xyz"].size(0)

            # updating spaces parameters
            self._gs_models[name].points_N = points_N
            self._gs_models[name].optimizer = new_optimizer
            self._gs_models[name].xyz = nn.Parameter(new_params["xyz"])
            self._gs_models[name].scales = nn.Parameter(new_params["scales"])
            print(new_params["xyz"].size(), new_params["scales"].size())
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
        padded_grads = torch.zeros(gs.xyz.size(0), 1).to(self.device)
        padded_grads[:grads.size(0)] = grads

        mask_grad = (padded_grads.squeeze() >= self.gradient_threshold)
        mask_size = (gs.scales.max(dim=-1).values <= self.percent_dense * self.cams_extent)
        mask = (mask_grad | mask_size)

        self._update_gs_model({
            "xyz": gs.xyz[mask],
            "scales": gs.scales[mask],
            "quats": gs.quats[mask],
            "opacities": gs.opacities[mask],
            "features_dc": gs.features_dc[mask],
            "features_rest": gs.features_rest[mask]
        }, name, "add")
    
    def _densify_and_split(self, name: str, grads: torch.Tensor) -> None:
        gs = self._gs_models[name]
        padded_grads = torch.zeros(gs.xyz.size(0), 1).to(self.device)
        padded_grads[:grads.size(0)] = grads 
       
        mask_grad = (padded_grads.squeeze() >= self.gradient_threshold)
        mask_size = (gs.scales.max(dim=-1).values >= self.percent_dense * self.cams_extent)
        mask = (mask_grad | mask_size)

        scales = gs.scales[mask].repeat(self.split_parts, 1)
        sampled_trans = torch.normal(mean=torch.zeros_like(scales), std=scales).to(self.device)
        Rs = build_rotation(gs.quats[mask]).repeat(self.split_parts, 1, 1)
        xyz = torch.bmm(Rs, (sampled_trans + gs.xyz[mask].repeat(self.split_parts, 1))[..., None]).squeeze()
        del sampled_trans, Rs, padded_grads
        # print(f"split info: {scales.size()}, {xyz.size()}")

        self._update_gs_model({
            "xyz": xyz,
            "scales": scales / (0.8 * self.split_parts),
            "quats": gs.quats[mask].repeat(self.split_parts, 1),
            "opacities": gs.opacities[mask].repeat(self.split_parts, 1),
            "features_dc": gs.features_dc[mask].repeat(self.split_parts, 1, 1),
            "features_rest": gs.features_rest[mask].repeat(self.split_parts, 1, 1)
        }, name, "add")

    def update_state(
        self, 
        update_filters: Optional[Dict[str, TensorType["Npoints"]]]=None,
        max_radii2D: Optional[TensorType["Npoints", "2DRadii"]]=None
    ) -> None:
        for (k, gs) in self._gs_models.items():
            gs = self._gs_models[k]
            grads = gs.xyz.grad
            assert (grads is not None), \
            ("gradients are NaN, you tried to run densification before backward()")
            if grads.isnan().any():
                warn(f" [NaN VALUES IN GRADIENTS: {k}]")
                grads[grads.isnan()] = 0.0
            if self._check_step(k):
                print(f"POINTS BEFORRE DENSIFICATION_[{k}]: {gs.xyz.size()}")
                grads_accumulated = (self._gradient_accums[k] / self._denoms[k])
                self._densify_and_split(k, grads_accumulated)
                self._densify_and_clone(k, grads_accumulated)

                mask_op = (gs.opacities < self.opacity_threshold).squeeze()
                mask_ext = (gs.scales.max(dim=-1).values > 0.1 * self.cams_extent)
                print(gs.opacities.size(), gs.scales.size())
                mask = (mask_op | mask_ext)
                if max_radii2D is not None:
                    mask_radii = (max_radii2D > self.max_screen_size)
                    mask = (mask | mask_radii)
                
                self._update_gs_model(mask, k, "prune")
                self._steps_schedule[k] += 1
                torch.cuda.empty_cache()
                print(f"POINTS AFTER DENSIFICATION_[{k}]: {gs.xyz.size()}")
            
            else:
                if update_filters is None:
                    filter = torch.ones_like(self._gradient_accums[k]).to(torch.bool)
                else:
                    assert (k in update_filters), \
                    (f"unknown filter name: {k}")
                    filter = update_filters[k]

                grads_norm = torch.linalg.norm(grads, dim=-1, keepdim=True)
                self._gradient_accums[k][filter] += grads_norm[filter]
                self._denoms[k][filter] += 1
                self._steps_schedule[k] += 1