import torch 
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from ..utils.projections import (
    make_mri_views,
    make_transform
)
from torch.utils.data import Dataset, DataLoader
from mne.io import read_raw_brainvision
from ..scene.gaussian_model import GaussianModel
from ..types import *


@dataclass
class ViewMixerDatasetConfig:
    source: Optional[str]=None
    # mri -> gs convertion parameters
    voxel_size: Optional[Tuple[int, int, int]]=(100, 100, 100)
    pts_scale: Optional[float]=10.0
    min_opacity_quant: Optional[float]=0.32
    max_opacity_quant: Optional[float]=0.64
    dst_coeff: Optional[float]=0.0
    recording_ms: Optional[int]=4 # [4ms, 14ms]
    stimul_types: Optional[List[str]]=field(default_factory=lambda: ["rest", "spTMS"]) # [rest, "spTMS"]
    f_min: Optional[float]=0.0
    f_max: Optional[float]=100.0
    target_freq_size: Optional[Tuple[int, int]]=(224, 448)

class ViewMixerDataset(Dataset):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        super(ViewMixerDataset, self).__init__()

    def load_mri(self, source: str) -> None:
        source_n = os.path.basename(source)
        source = os.path.join(source, f"/semi-mri/anat/{source_n}_semi-mri_Tw1.nii.gz")
        assert (os.path.exists(source)), \
        (f"couln't find MRI data in at location: {source}")

        self.base_mri_gs = GaussianModel("base_mri")
        self.base_mri_gs.load_nifti(source,
                                    self.cfg.pts_scale,
                                    self.cfg.voxel_size,
                                    self.cfg.max_opacity_quant,
                                    self.cfg.min_opacity_quant,
                                    self.cfg.dst_coeff)
        self.mri_slice_views = make_mri_views(self.cfg.voxel_size, self.cfg.pts_size)

    def read_sensors(self, source: str) -> None:
        
        source_n = os.path.basename(source)
        sensor_type = f"ses-async{self.cfg.recording_ms}ms"
        annots_f = os.path.join(source, f"{sensor_type}/{source_n}_{sensor_type}_scans.tsv")
        locations_f = os.path.join(source, f"{sensor_type}/eeg/{source_n}_{sensor_type}_electrodes.tsv")
        assert os.path.exists(locations_f), \
        (f"couldn't find sensors locations annotation file at location: {locations_f}")
        assert os.path.exists(annots_f), \
        (f"couldn't load records annotations from: {annots_f}")
        
        locations_df = pd.read_csv(locations_f, sep="\t").iloc[:, :-1].dropna()
        names = locations_df["name"].tolist()
        locations = locations_df.iloc[:, 1:5].to_numpy()
        self.channel_index2location = {k_ch: xyz for (k_ch, xyz) in zip(names, locations)}

        annots_df = pd.read_csv(annots_f, sep="\t")
        for f_name in annots_df["filename"]:
            stimul_type = f_name[f_name.find("task") + 5: f_name.find("run") - 1]
            print(stimul_type)
            if stimul_type not in self.cfg.stimul_types:
                continue

            eeg_file = os.path.join(source, sensor_type, f_name)
            raw = read_raw_brainvision(eeg_file)
            print(int(raw.info["sfreq"] * 2))
            pcd = raw.compute_psd(average=None, 
                                fmax=self.cfg.f_max, 
                                n_fft=int(raw.info["sfreq"] * 4),
                                n_overlap=int(raw.info["sfreq"] * 3.8))
            signals = torch.Tensor(raw.get_data())
            Sxx = torch.Tensor(pcd.get_data())[:, :13, :]
            print(Sxx.size())
            Sxx = F.interpolate(Sxx[None], 
                                self.cfg.target_freq_size,
                                mode="bilinear").squeeze()
            
            import matplotlib.pyplot as plt
            plt.style.use("dark_background")

            _, axis = plt.subplots()
            axis.imshow(Sxx[0])
            plt.show()
            break


            
            
            



if __name__ == "__main__":

    source = "/media/ram/T71/ds004024-download/sub-CON001"
    config = ViewMixerDatasetConfig(f_max=100)
    dataset = ViewMixerDataset(config)
    dataset.read_sensors(source)
    