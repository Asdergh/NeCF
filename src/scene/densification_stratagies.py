import torch 
import torch.nn as nn
import abc 
from torch.optim import (Adam, Optimizer)
from warnings import warn
from .gaussian_model import GaussianModel
from ..utils.general_utils import build_rotation
from ..types import *



class DensificationBase(abc.ABC):
    def __init__(self) -> None:
        self._gs_models: Dict[str, GaussianModel] = {}
    
    @abc.abstractmethod
    def add_gs_model(self, gs: GaussianModel) -> None:
        """
        Docstring for add_gs_model

        This method adds new gaussian spaces into _gs_models collection. 
        Gradients of gaussian models contained in this collection will 
        be changed, so be sure of cheking is there is a right space was added or not!!!

        :param self: Description
        :param gs: GaussianModel instance with already initialized arguments
        :type gs: GaussianModel
        """
        raise NotImplementedError("""add_gs_models method have different buffers to fill
                                  depending onn Stratagy type""")

    @abc.abstractmethod
    def update_gs_model(self) -> None:
        """
        Docstring for update_gs_model
        This function initiinitilizez distributes the tensors values 
        for all specified gs_model by its name and mask. 
        :param self: Description
        """
        raise NotImplemented(f"""Use update_gs_models to distribute 
                             tensots values for all gs_model params!!!""")
    @abc.abstractmethod
    def update_state(self) -> None:
        """
        Docstring for update_state
        This is the main function for densification in with stratagy is implemented.
        Be sure that in your custom densification stratagy you didnt forgot to implement 
        this function !!!
        :param self: Description
        """
        raise NotImplemented("""This is the main function for every densification stratagy.
                             without it you couln't optimize your splats""")

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

        # updating densifiaction managing properties


# --------------------------------------------------------------------------------------------------------
# TODO find a way to implement two operations in DefaultStratagy to nfrshare the principales of ABC class
# self._gradient_accums[name] = torch.zeros(new_params["xyz"].size(0), 1).to(self.device)
# self._denoms[name] = torch.zeros(new_params["xyz"].size(0), 1).to(self.device)
# --------------------------------------------------------------------------------------------------------
class DefaultStratagy(DensificationBase):
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
        super(DefaultStratagy, self).__init__()
        # attributes to manage densification procedure
        
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

    def add_gs_model(self, gs: GaussianModel) -> None:        
        self._gs_models.update({gs.name: gs})
        self._gradient_accums.update({gs.name: torch.zeros(gs.points_N, 1).to(self.device)})
        self._denoms.update({gs.name: torch.zeros(gs.points_N, 1).to(self.device)})
        self._steps_schedule.update({gs.name: 0})
    
    def update_gs_model(self, name, update_pkg, mode):
        (new_params, new_optimizer) = self._update_optimizer(name, input_pkg=update_pkg, mode=mode)
        points_N = new_params["xyz"].size(0)

        # updating spaces parameters
        self._gs_models[name].points_N = points_N
        self._gs_models[name].optimizer = new_optimizer
        self._gs_models[name].xyz = nn.Parameter(new_params["xyz"])
        self._gs_models[name].scales = nn.Parameter(new_params["scales"])
        self._gs_models[name].quats = nn.Parameter(new_params["quats"])
        self._gs_models[name].opacities = nn.Parameter(new_params["opacities"])
        self._gs_models[name].features_dc = nn.Parameter(new_params["features_dc"])
        self._gs_models[name].features_rest = nn.Parameter(new_params["features_rest"])

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

        self.update_gs_model({
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

        self.update_gs_model({
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
                
                self.update_gs_model(mask, k, "prune")
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

class BoundsRestrictionStratagy(DensificationBase):
    def __init__(self) -> None:
        super(BoundsRestrictionStratagy, self).__init__()
        self._rad: Dict[str, float] = {}
        self._theta: Dict[str, float] = {}
        self._phi: Dict[str, float] = {}
        
    def add_gs_model(self, gs_model: GaussianModel, **kwargs) -> None:

        if kwargs["theta_max"] is None \
            or kwargs["phi_max"] is None \
            or kwargs["rad_max"] is None:
            warn(f"""There are no bounderies for gs_model: {gs_model.name}.
                 You can add them with add_bounds(name: str, theta: float, phi: float, rad: float). 
                 Be sure that name is in self._gs_models: Dict[str, GaussianModel] 
                 registry""")
        
        self._gs_models[gs_model.name] = gs_model
        self._theta[gs_model.name] = kwargs["theta_max"]
        self._phi[gs_model.name] = kwargs["phi_max"]
        self._rad[gs_model.name] = kwargs["rad_max"]
    
    def add_bounds(self, name: str, theta: float, phi: float, rad: float) -> None:
        assert (name in self._gs_models), \
        (f"The is not gs_model with name: {name} in self._gs_models buffer")
        self._theta[name] = theta
        self._phi[name] = phi
        self._rad[name] = rad
    
    def check_model(self, name: str) -> None:
        gs = self._gs_models[name]
        xyz = gs.xyz

        norms = torch.linalg.norm(gs.xyz, dim=-1)
        gs_theta = torch.arccos(xyz[:, -1] / norms)
        gs_phi = torch.arctan(xyz[:, 1] / xyz[:, 0])

        action = {
            (True, True, True): lambda t, p, r: (torch.abs(t) <= self._theta[name]) 
                                                & (torch.abs(p) <= self._phi[name]) \
                                                & (r <= self._rad[name]),
            (True, True, False): lambda t, p, r: (torch.abs(t) <= self._theta[name]) \
                                                & (torch.abs(p) <= self._phi[name]),
            (True, False, True): lambda t, p, r: (torch.abs(t) <= self._theta[name]) \
                                                & (r <= self._rad[name]),
            (False, True, True): lambda t, p, r: (torch.abs(p) <= self._phi[name]) \
                                                & (r <= self._rad[name]),
            (True, False, False): lambda t, p, r: (torch.abs(t) <= self._theta[name]),
            (False, True, False): lambda t, p, r: (torch.abs(p) <= self._phi[name]),
            (False, False, True): lambda t, p, r: (r <= self._rad[name])
        }
        return action(gs_theta, gs_phi, norms)

    def update_gs_model(self, name: str, mask: torch.Tensor) -> None:
        gs = self._gs_models[name]
        if hasattr(gs, "optimizer"):
            (new_tensors, optimizer) = self._update_optimizer(name, mask, "prune")
            gs.optimizer = optimizer
        else:
            new_tensors = {
                "xyz": gs.xyz[mask],
                "scales": gs.scales[mask],
                "quats": gs.quats[mask],
                "opacities": gs.opacities[mask],
                "features_dc": gs.features_dc[mask],
                "features_rest": gs.features_rest[mask]
            }
        for (k, v) in new_tensors:
            setattr(gs, k, nn.Parameter(v.required_grad_(True)))
        self._gs_models[name] = gs
        
        
    def update_state(self):
        for k in self._gs_models.items():
            if self._theta[k] is None \
            and self._phi[k] is None \
            and self._rad[k] is None:
                warn(f"""There is not bounderies for gs_model: {k}.
                     Is this behaviour if not expected try to use 
                     add_bounderies(name: str, theta: float, phi: theta, rad: float)
                     or try to pass some bounderies at initialization with 
                     add_gs_model(gs_model: GaussianModel, theta: float, phi: float, rad: float)""")
                continue
            mask = self.check_model(k)
            self.update_gs_model(k, mask)
        
        
