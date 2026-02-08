import torch 
import numpy as np
import math
import viser 
import rerun as rd
import time
import os
from warnings import warn
from torch.utils.tensorboard import SummaryWriter
from ..types import *
from ..scene.gaussian_model import GaussianModel
from ..utils.projections import Rmat2quat
from scipy.spatial.transform import Rotation as R


class Logger:
    def __init__(self, source: Optional[str]=None) -> None:
        self.source = source
        self._server = viser.ViserServer()
        if self.source is None:
            warn(f"""Coundn't find specified location " \
            to load the data: {self.source}. 
            Files will be loged into root directory of the project.
            If this behaviour if unexetable try to pass existing file!!!""")
            self.source = "logs"
            if not os.path.exists(self.source):
                os.mkdir(self.source)
        
        # gaussian splatting data to log with viser
        self.gs_source = os.path.join(self.source, "ply_files")
        if not os.path.exists(self.gs_source): os.mkdir(self.gs_source)
        self._gs_maps: Dict[str, GaussianModel] = {}

        # utils to load tensorboard logs
        self.tb_source = os.path.join(self.source, "tensorboard")
        if not os.path.exists(self.gs_source): os.mkdir(self.gs_source)
        self._writer_tb = SummaryWriter(self.tb_source)

        self.get_tpr2xyz = lambda Twc, t, p, r: (Twc @ torch.Tensor([
            r * math.cos(t),
            -r * math.sin(t) * math.sin(p),
            r * math.cos(t) * math.sin(p), 1
        ]))[:-1]
    
    def add_sensor_space(self, gs: GaussianModel) -> None:
        self._gs_maps[gs.name] = gs

    def log_sensor_bounds2viser(
        self, 
        loging_path: str,
        Transform: torch.Tensor, 
        theta_max: Optional[float],
        phi_max: float,
        rad_max: float,
        depth: float
    ):
        
        c = Transform[:3, 3]
        shift = (depth * Transform[:3, 0])
        center = self.get_tpr2xyz(Transform, 0, 0, rad_max)
        up_l = self.get_tpr2xyz(Transform, -theta_max, -phi_max, rad_max)
        up_r = self.get_tpr2xyz(Transform, theta_max, -phi_max, rad_max)
        down_l = self.get_tpr2xyz(Transform, -theta_max, phi_max, rad_max)
        down_r = self.get_tpr2xyz(Transform, theta_max, phi_max, rad_max)

        colors = np.random.randint(0, 255, (3, ))
        base_segments = torch.stack([
                torch.stack([c, up_l], dim=0),
                torch.stack([c, up_r], dim=0),
                torch.stack([c, down_l], dim=0),
                torch.stack([c, down_r], dim=0),
                torch.stack([up_l, center], dim=0),
                torch.stack([up_r, center], dim=0),
                torch.stack([down_l, center], dim=0),
                torch.stack([down_r, center], dim=0),
                torch.stack([up_l, up_r], dim=0),
                torch.stack([up_l, down_l], dim=0),
                torch.stack([down_l, down_r], dim=0),
                torch.stack([down_r, up_r], dim=0),
            ], dim=0) + shift
        self._server.scene.add_line_segments(loging_path, base_segments, colors)

    def log_sensor2viser(self, name: str) -> None:
        gs = self._gs_maps[name]
        sensor = gs.sensor
        Twc = sensor.Transform
        log_path = f"sensors/{gs.name}_channel"
        splats_p = f"{log_path}/field"
        sensors_p = f"{log_path}/sensors_bounds"
        for idx, (theta, phi, rad, depth) in enumerate(zip(sensor.theta_max,
                                                 sensor.phi_max,
                                                 sensor.rad_max,
                                                 sensor.depths)):
            
            bbox_p = f"{sensors_p}/sensor{idx}"
            bbox_local_xyz_p = f"{sensors_p}/local_xyz"
            bbox_box_p = f"{bbox_p}/box_segments"
            bbox_path_p = f"{bbox_p}/path_betweetn_o_and_sensor_{idx}"

            c = Twc[:3, 3]
            shift = (depth * Twc[:3, 0])
            theta = math.radians(theta)
            phi = math.radians(phi)
            color = 255 * np.ones(3)
            
            self.log_sensor_bounds2viser(bbox_box_p, Twc, theta, phi, rad, depth)
            self._server.scene.add_line_segments(bbox_path_p, torch.stack([c, c + shift], dim=0)[None], color)
            self._server.scene.add_frame(bbox_local_xyz_p, wxyz=R.from_matrix(Twc[:3, :3]).as_quat(scalar_first=True), 
                                        position=Twc[:3, 3], 
                                        axes_length=10.0,
                                        axes_radius=0.2)

    def run_view(self, sleep: Optional[float]=10.0) -> None:
        self._server.scene.set_background_image(np.zeros((100, 100, 3)))
        while True:
            time.sleep(sleep)



if __name__ == "__main__":
    from ..utils.projections import make_transform
    from ..scene.gaussian_model import (Sensor, GaussianModel)

    logger = Logger(source=None)
    c = torch.normal(0, 100, (5, 3))
    t = torch.zeros(3)
    Twcs = make_transform(c, t)
    for idx, Twc in enumerate(Twcs):
        sensor = Sensor(
            theta_max=[23, 23, 23],
            phi_max=[65, 65, 65],
            rad_max=[23, 23, 23],
            depths=[100, 75, 55],
            Transform=Twc
        )
        gs = GaussianModel(name=f"gs_model_{idx}", sensor=sensor)
        logger.add_sensor_space(gs)
        logger.log_sensor2viser(gs.name)
    
    logger.run_view()
    




            

            
            

            
        

        
        


