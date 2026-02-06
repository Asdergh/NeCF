import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R
from ..types import *
plt.style.use("dark_background")






def quat2Rmat(q: Union[torch.Tensor, np.ndarray], in_format="xyzw") -> np.ndarray:
    
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()    

    if (q.ndim == 1):
        batched = False
        N = 1
        q = q.reshape(1, 4)
    elif (q.ndim == 2 and q.shape[-1] == 4):
        N = q.shape[0]
        batched = True
    else:
        raise ValueError(f"q is not quaternion vector or batch of vectors. Size: {q.shape}")
        
    if in_format == "xyzw":
        x = q[:, 0]
        y = q[:, 1]
        z = q[:, 2]
        w = q[:, 3]
    else:
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        w = q[:, 0]
    
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    Rmat_batched = np.zeros((N, 3, 3))
    Rmat_batched[:, 0, 0] = 1 - 2*(y2 + z2)
    Rmat_batched[:, 0, 1] = 2*(xy - wz)
    Rmat_batched[:, 0, 2] = 2*(xz + wy)

    Rmat_batched[:, 1, 0] = 2*(xy + wz)
    Rmat_batched[:, 1, 1] = 1 - 2*(x2 + z2)
    Rmat_batched[:, 1, 2] = 2*(yz - wx)

    Rmat_batched[:, 2, 0] = 2*(xz - wy)
    Rmat_batched[:, 2, 1] = 2*(yz + wx)
    Rmat_batched[:, 2, 2] = 1 - 2*(x2 + y2)

    return (Rmat_batched[0] if not batched else Rmat_batched)

def Rmat2quat(Rmat: Union[torch.Tensor, np.ndarray], out_format="xyzw") -> np.ndarray:
    
    if isinstance(Rmat, torch.Tensor):
        Rmat = Rmat.detach().cpu().numpy() 

    if Rmat.ndim == 3:
        batched = True
        N = Rmat.shape[0]
    else:
        N = 1
        batched = False
        Rmat = Rmat.reshape(1, 3, 3)

    Tr = np.diagonal(Rmat, axis1=1, axis2=2).sum(axis=1)
    q_result = np.zeros((N, 4))
    max_diag = np.stack([Rmat[:, 0, 0], Rmat[:, 1, 1], Rmat[:, 2, 2]], axis=-1).max(axis=1)
    
    where_tr_pos = (Tr > 0)
    if where_tr_pos.sum() != 0.0:
        w = np.sqrt(1 + Tr) / 2
        x = (Rmat[where_tr_pos, 2, 1] - Rmat[where_tr_pos, 1, 2]) / (4 * w + 1e-6)
        y = (Rmat[where_tr_pos, 0, 2] - Rmat[where_tr_pos, 2, 0]) / (4 * w + 1e-6)
        z = (Rmat[where_tr_pos, 1, 0] - Rmat[where_tr_pos, 0, 1]) / (4 * w + 1e-6)
        q_result[where_tr_pos] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    where_r11_max = (Rmat[:, 0, 0] == max_diag)
    if where_r11_max.sum() != 0.0:
        x = (np.sqrt(1 + 2 * Rmat[where_r11_max, 0, 0] - Tr[where_r11_max])) / 2
        w = (Rmat[where_r11_max, 2, 1] - Rmat[where_r11_max, 1, 2]) / (4 * x + 1e-6)
        y = (Rmat[where_r11_max, 0, 1] + Rmat[where_r11_max, 1, 0]) / (4 * x + 1e-6)
        z = (Rmat[where_r11_max, 0, 2] + Rmat[where_r11_max, 2, 0]) / (4 * x + 1e-6)
        q_result[where_r11_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r22_max = (Rmat[:, 1, 1] == max_diag)
    if where_r22_max.sum() != 0.0:
        y = (np.sqrt(1 + 2 * Rmat[where_r22_max, 1, 1] - Tr[where_r22_max])) / 2
        w = (Rmat[where_r22_max, 0, 2] - Rmat[where_r22_max, 2, 0]) / (4 * y + 1e-6)
        x = (Rmat[where_r22_max, 0, 1] + Rmat[where_r22_max, 1, 0]) / (4 * y + 1e-6)
        z = (Rmat[where_r22_max, 1, 2] + Rmat[where_r22_max, 2, 1]) / (4 * y + 1e-6)
        q_result[where_r22_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r33_max = (Rmat[:, 2, 2] == max_diag)
    if where_r33_max.sum() != 0.0:
        z = (np.sqrt(1 + 2 * Rmat[where_r33_max, 2, 2] - Tr[where_r33_max])) / 2
        w = (Rmat[where_r33_max, 1, 0] - Rmat[where_r33_max, 0, 1]) / (4 * z + 1e-6)
        x = (Rmat[where_r33_max, 0, 2] - Rmat[where_r33_max, 2, 0]) / (4 * z + 1e-6)
        y = (Rmat[where_r33_max, 1, 2] - Rmat[where_r33_max, 2, 1]) / (4 * z + 1e-6)
        q_result[where_r33_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    q_norms = np.linalg.norm(q_result, axis=-1, keepdims=True)
    q_result = (q_result / q_norms)
    return (q_result[0] if not batched else q_result)

def pos_embedding(x: torch.Tensor, L: int) -> torch.Tensor:
    """positional encoding function from NeRF"""
    scales = torch.Tensor([(2 ** i) * torch.pi for i in range(L)])
    x = (x[:, None, :] * scales[None, :, None])
    x[0::2] = torch.sin(x[0::2])
    x[1::2] = torch.cos(x[1::2])
    return x

def cart2cylinder(x: torch.Tensor) -> torch.Tensor:
    """convert (x, y, z) carteziation coordinates int cylinderycal (phi, rho, z)"""
    phi = torch.atan(x[..., 1] / x[..., 0])
    rho = torch.sqrt((x[..., 0] * x[..., 0]) + (x[..., 1] * x[..., 1]))
    z = x[..., -1]
    return torch.stack([phi, rho, z], dim=-1)
    
def cylinder2cart(x: torch.Tensor) -> torch.Tensor:
    """convert (x, y, z) carteziation coordinates int cylinderycal (phi, rho, z)"""
    xc = x[..., 1] * torch.cos(x[..., 0])
    yc = x[..., 1] * torch.sin(x[..., 0])
    zc = x[..., -1]
    return torch.stack([xc, yc, zc], dim=-1)

def get_cylinder_dirs(
    angles_max: int=10,
    heights_max: int=1,
    max_height: float=1.5,
    rho: float=5.0
) -> TensorType["Npoints", "XYZ"]:
    
    angles = torch.linspace(-torch.pi, torch.pi, angles_max)
    heights = torch.linspace(0, max_height, heights_max)
    xyz = torch.cat([
        torch.cat([
            rho * torch.sin(angles)[:, None],
            rho * torch.cos(angles)[:, None], 
            height.repeat(angles_max, 1)
        ], dim=-1)
        for height in heights
    ], dim=0)
    xyz = cylinder2cart(xyz)
    return xyz

def cross_check(r: torch.Tensor, t: torch.Tensor):
    print(r.size(), t.size())
    cross_map = torch.linalg.cross(t, r)
    cross_map /= torch.linalg.norm(cross_map, dim=-1, keepdim=True)
    cross_map = torch.linalg.norm(cross_map, dim=-1)
    map = torch.where(cross_map > 0, True, False)
    return map

def make_transform(c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Docstring for make_transform
    All the Transformations is generated with assumption that all 
    \vec{c} points are highter the global_xyz points and placed on
    sphere surface thats covers the MRI in NeCFf coordinates convention

    :param c: The location of local coordinate system
    :type c: torch.Tensor
    :param t: The point of direction in with Neural Fields woulbd be generated
    :type t: torch.Tensor
    :return: The Transformations to the local coordinate system at c alond direction to t
    :rtype: Tensor
    """
    N = c.size(0)
    if t.ndim == 1:
        t = t.repeat(N, 1)

    forward = (t - c) / torch.linalg.norm(t - c, dim=-1, keepdim=True)
    world_xyz = torch.eye(3)
    map_uz = cross_check(forward, world_xyz[2][None])
    map_uy = torch.logical_and(~map_uz, cross_check(forward, world_xyz[1][None]))    
    left = torch.zeros(N, 3)
    up = torch.zeros(N, 3)

    Nuz = torch.sum(map_uz)
    if Nuz:
        dy = torch.linalg.cross(world_xyz[2][None], forward[map_uz])
        dy /= torch.linalg.norm(dy, dim=-1, keepdim=True)
        dz = torch.linalg.cross(forward[map_uz], dy)
        dz /= torch.linalg.norm(dz, dim=-1, keepdim=True)
        left[map_uz] = dy
        up[map_uz] = dz
    
    Nuy = torch.sum(map_uy)
    if Nuy:
        dz = torch.cross(forward[map_uy], world_xyz[1][None])
        dz /= torch.linalg.norm(dz, dim=-1, keepdim=True)
        dy = torch.linalg.cross(dz, forward[map_uy])
        dy /= torch.linalg.norm(dy, dim=-1, keepdim=True)
        left[map_uy] = dy
        up[map_uy] = dz
    
    Transform = torch.zeros(c.size(0), 4, 4)
    Transform[:, :3, 0] = forward
    Transform[:, :3, 1] = left
    Transform[:, :3, 2] = up

    Transform[:, :3, 3] = c
    Transform[:, 3, 3] = 1.0
    return Transform


def make_mri_views(grid_size: Tuple[int, int, int], pos_factor: float=1.0):

    world_axis = torch.eye(3)
    
    lenghts = lambda i: (torch.arange(grid_size[i]) * pos_factor)
    f_p = world_axis[0].repeat(grid_size[0], 1) * lenghts(0)[:, None]
    b_p = -f_p
    r_p = world_axis[1].repeat(grid_size[1], 1) * lenghts(1)[:, None]
    l_p = -r_p
    u_p = world_axis[2].repeat(grid_size[2], 1) * lenghts(2)[:, None]
    d_p = -u_p
    c = torch.cat([f_p, b_p, 
                   r_p, l_p, 
                   u_p, d_p], dim=0)
    t = torch.zeros(1, 3)
    Transforms = make_transform(c, t)
    return Transforms
    

def show_local_global_connections(name, c, t):
    rr.log(f"{origin}/local_global_directions{name}",
           rr.Arrows3D(
               vectors=c,
               origins=t
           ))
def show_world_axis(name, Twc) -> None:
    rr.log(f"{origin}/{name}",
           rr.Arrows3D(
               vectors=[Twc[:3, 0], Twc[:3, 1], Twc[:3, 2]],
               origins=Twc[:3, 3][None],
               colors=[
                   [255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255]
               ]
           )) 

def show_points(name, xyz) -> None:
    rr.log(f"{origin}/PointsXYZ{name}",
           rr.Points3D(
               positions=xyz,
               radii=0.03
           ))






if __name__ == "__main__":
    
    import rerun as rr
    
    origin = "test"
    rr.init(origin, spawn=True)
    # c = torch.normal(0, 5, (100, 3))
    Ts1 = make_mri_views((100, 100, 10))
    points = torch.normal(0, 0.3, (100, 3))
    t = torch.zeros(3)
    # show_local_global_connections("1", c, t)
    
    # Ts1 = make_transform(c, t[0])
    for idx, Twc in enumerate(Ts1):
        show_world_axis(f"Frame0{idx}", Twc)
        pts = points
        trans = torch.Tensor([3.0, 0.0, 0.0])
        # pts[:, 0] -= 
        pts = (Twc[:3, :3] @ points.T).T
        pts += Twc[:3, 3]
        trans = (Twc[:3, :3] @ trans)
        trans += Twc[:3, 3]
        pts += (Twc[:3, 0] *2)
        show_points(name=idx, xyz=pts)
    

    
    

        
    

    
   
    

    