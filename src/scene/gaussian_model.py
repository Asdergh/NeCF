import torch 
import torch.nn as nn
from torch.optim import (Adam, Optimizer)
from warnings import warn
from ..utils import (build_rotation)
from ..types import *


@dataclass
class GaussianModel:
    xyz: TensorType["Ncenters", "XYZ"]
    features_dc: TensorType["Ncenters", "FEATURES", "RGB"]
    features_rest: TensorType["Ncenters", "FEATURES", "RGB"]
    opacities: TensorType["Ncenters", "Value"]
    scales: TensorType["Ncenters", "XYZ"]
    quats: TensorType["Ncenters", "XYZW"]
    
    
    def setup_optimizer(self, opt_cfg) -> None:
        params = [
            {"params": [self.xyz], "lr": opt_cfg.xyz_lr_rate, "name": "xyz"},
            {"params": [self.scales], "lr": opt_cfg.scales_lr_rate, "name": "scales"},
            {"params": [self.quats], "lr": opt_cfg.quats_lr_rate, "name": "quats"},
            {"params": [self.opacities], "lr": opt_cfg.opacities_lr_rate, "name": "opacities"},
            {"params": [self.features_dc], "lr": opt_cfg.features_dc_lr_rate, "name": "features_dc"},
            {"params": [self.features_rest], "lr": opt_cfg.features_rest_lr_rate, "name": "features_rest"}
        ]
        self.optimizer = Adam(params)
        if opt_cfg.optimize_cams:
            self.cams_optimizer = Adam(lr=opt_cfg.cams_lr_rate)
    

class DensificationStrategyDefault:
    def __init__(
        self,
        cams_extent: float,
        max_screen_size: float,
        opacity_threshold: Optional[float]=0.2,
        gradient_threshold: Optional[float]=None,
        split_parts: Optional[int]=2,
        percent_dense: Optional[float]=0.1
    ) -> None:
        
        # attributes to manage densification procedure
        self._gs_models: Dict[str, GaussianModel] = {}
        self._gradient_accums: Dict[str, TensorType["Ncenters", "XYZ"]] = {}
        self._denoms = torch.empty(0)
        self.cams_extent = cams_extent
        self.max_screen_size = max_screen_size
        self.opacity_threshold = opacity_threshold
        self.gradient_threshold = gradient_threshold
        self.split_parts = split_parts
        self.percent_dense = percent_dense


    def _update_optimizer(
        self, 
        gs_model_name: str,
        input_pkg: Dict[str, torch.Tensor],
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
            assert (input_pkg is not None \
                        and (group["name"] in input_pkg) \
                        and (input_pkg[group["name"]] is not None)), \
                (f"""to delete gaussian from optimizer you must provide correct input_pkg. 
                Current value for this arg: {input_pkg} """)
            if mode in ["delete", "purge", "prune"]:
                valid_mask = input_pkg[group["name"]]
                new_params = group["params"][0][valid_mask]
            elif mode in ["add", "cat", "stack"]:
                tensor2add = input_pkg[group["name"]]
                new_params = torch.cat([group["params"][0], tensor2add], dim=0)
            else:
                raise ValueError(f"unknow mode: {mode}")

            stored_state = optimizer.states.get(group["params"][0], None)
            optimizable_tensors.update({group["name"]: new_params})
            if stored_state is not None:
                if mode in ["delete", "purge", "prune"]:
                    stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                elif mode in ["add", "cat", "stack"]:
                    stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], tensor2add], dim=0)
                    stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], tensor2add], dim=0)
                    
                del optimizer.states[group["params"][0]]
                optimizer.states.update({group["params"][0]: stored_state})

            return (optimizable_tensors, optimizer)

        
    def _update_optimizers(self, update_pkg: Dict[str, Any], mode: str) -> None:
        for (k, pkg) in update_pkg.items():
            (new_params, new_optimizer) = self._update_optimizer(k, input_pkg=pkg, mode=mode)
            self._gs_models[k].optimizer = new_optimizer
            self._gs_models[k].xyz = nn.Parameter(new_params["xyz"])
            self._gs_models[k].scales = nn.Parameter(new_params["scales"])
            self._gs_models[k].quats = nn.Parameter(new_params["quats"])
            self._gs_models[k].opacities = nn.Parameter(new_params["opacities"])
            self._gs_models[k].features_dc = nn.Parameter(new_params["features_dc"])
            self._gs_models[k].features_rest = nn.Parameter(new_params["features_rest"])
        

    def _densify_and_clone(self, states_pkg: Dict[str, torch.Tensor]):
        update_pkg = {}
        for (k, gs_model) in self._gs_models.items():
            grads = states_pkg[k]
            mask = torch.where(torch.linalg.norm(grads, dim=-1) <= self.gradient_threshold, True, False)
            mask = torch.logical_or(mask, torch.where(
                gs_model.scales <= self.percent_dense * self.cams_extent, 
                True, False
            ))
            update_pkg.update({k: {
                "xyz": gs_model.xyz[mask],
                "scales": gs_model.scales[mask],
                "quats": gs_model.quats[mask],
                "opacities": gs_model.opacities[mask],
                "features_dc": gs_model.features_dc[mask],
                "features_rest": gs_model.features_rest[mask]
            }})
        self._update_optimizers(update_pkg, "add")
    
    def _densify_and_split(self,  states_pkg: Dict[str, torch.Tensor]) -> None:
        update_pkg = {}
        for (k, gs_model) in self._gs_models.items():
            mask = torch.where(torch.linalg.norm(states_pkg[k], dim=-1) <= self.gradient_threshold, True, False)
            mask = torch.logical_or(mask, torch.where(
                gs_model.scales >= self.percent_dense * self.cams_extent, 
                True, False
            ))

            scales = gs_model.scales[mask].repeat(self.split_parts, 1)
            sampled_trans = torch.normal(mean=torch.zeros_like(scales), std=scales)
            Rs = build_rotation(gs_model.quats[mask]).repeat(self.split_parts, 1, 1)
            xyz = torch.bmm(Rs, (sampled_trans + gs_model.xyz[mask].repeat(self.split_parts, 1)[..., None])).squeeze()
            update_pkg.update({k: {
                "xyz": xyz,
                "scales": scales / (0.8 * self.split_parts),
                "quats": gs_model.quats[mask].repeat(self.split_parts, 1),
                "opacities": gs_model.opacities[mask].repeat(self.split_parts, 1),
                "features_dc": gs_model.features_dc[mask].repeat(self.split_parts, 1, 1),
                "features_rest": gs_model.features_rest[mask].repeat(self.split_parts, 1, 1)
            }})
        self._update_optimizers(update_pkg, "add")

            
    def update_state(self, states: Dict[str, TensorType["Ncenters", "XYZ"]]) -> None:
        # checking NaN values
        for (k, grads) in states.items():
            if grads.isnan().any():
                warn(f" [NaN VALUES IN GRADIENTS: {k}]")
                grads[grads.isnan()] = 0.0
                states.update({k: grads})

        # applying densification procedures
        self._densify_and_split(states)
        self._densify_and_clone(states)

        # pruning gs models parameters and gradients states
        prune_masks = {}
        for (k, grads) in states.items():
            mask = torch.where(self._gs_models[k].opacities < self.opacity_threshold, False, True)
            mask = torch.logical_or(mask, torch.where(self._gs_models[k].scales.max(dim=-1) > self.max_screen_size, False, True))
            mask = torch.logical_or(mask, torch.where(self._gs_models[k].scales.max(dim=-1) > 0.1 * self.cams_extent, False, True))
            prune_masks.update({k: mask})

        self._update_optimizers(prune_masks, "prune")
        