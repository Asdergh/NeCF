import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R



line = np.linspace(0, 100).reshape(-1, 1).repeat(1, 3)
translation = np.array([1.0, 1.0, 0.0])
Rmat = R.from_rotvec(45.0 * np.array([1


