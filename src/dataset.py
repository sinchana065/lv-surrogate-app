import h5py, json, numpy as np, torch
from torch.utils.data import Dataset

class LVH5(Dataset):
    def __init__(self, h5_path, split_npz, split_key, stats_json):
        self.h5_path = h5_path
        self.idx = np.load(split_npz)[split_key]
        with open(stats_json) as r:
            self.S = json.load(r)

    def __len__(self): 
        return len(self.idx)

    def _z(self, x, mean, std):  # z-score
        return (x - mean) / (std + 1e-8)

    def _mm(self, x, lo, hi):    # min-max
        return (x - lo) / (hi - lo + 1e-8)

    def __getitem__(self, i):
        t = int(self.idx[i])
        with h5py.File(self.h5_path, "r") as f:
            sdf  = f["/inputs/sdf"][t]      # (H,W)
            hr   = f["/inputs/hr"][t]
            sbp  = f["/inputs/sbp"][t]
            dbp  = f["/inputs/dbp"][t]
            ph   = f["/inputs/phase"][t]
            u    = f["/labels/u"][t]
            v    = f["/labels/v"][t]
            p    = f["/labels/p"][t]

        # normalize fields
        sdf = self._z(sdf, self.S["sdf_mean"], self.S["sdf_std"])
        u   = self._z(u,   self.S["u_mean"],   self.S["u_std"])
        v   = self._z(v,   self.S["v_mean"],   self.S["v_std"])
        p   = self._z(p,   self.S["p_mean"],   self.S["p_std"])

        # params → [0,1]
        hr  = float(self._mm(hr,  self.S["hr_min"],  self.S["hr_max"]))
        sbp = float(self._mm(sbp, self.S["sbp_min"], self.S["sbp_max"]))
        dbp = float(self._mm(dbp, self.S["dbp_min"], self.S["dbp_max"]))

        # phase → sin/cos
        P = self.S["phase_period"]
        ang = 2*np.pi*(ph % P)/P
        ph_s, ph_c = float(np.sin(ang)), float(np.cos(ang))

        # broadcast scalars to maps
        H, W = sdf.shape
        def M(x): return np.full((H,W), x, dtype=np.float32)

        inp = np.stack([
          sdf.astype(np.float32),
          M(hr), M(sbp), M(dbp),
          M(ph_s), M(ph_c)
        ], axis=0)  # (6,H,W)

        tgt = np.stack([
          u.astype(np.float32),
          v.astype(np.float32),
          p.astype(np.float32)
        ], axis=0)  # (3,H,W)

        return torch.from_numpy(inp), torch.from_numpy(tgt)

