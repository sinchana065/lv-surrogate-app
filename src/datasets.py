from torch.utils.data import Dataset, DataLoader, random_split
import torch
import h5py
import os

class McWilliamsDataset(Dataset):
    def __init__(self, data_root="data", split="train"):
        """
        Loads synthetic left ventricle dataset from HDF5 file.
        Inputs: sdf (spatial field) + scalar values (dbp, hr, phase, sbp)
        Labels: u, v, p (flow field outputs)
        """
        path = os.path.join(data_root, "synthetic_lv_dataset_small.h5")

        with h5py.File(path, "r") as f:
            # Load scalar inputs
            dbp = f["inputs/dbp"][:]     # (N,)
            hr = f["inputs/hr"][:]       # (N,)
            phase = f["inputs/phase"][:] # (N,)
            sbp = f["inputs/sbp"][:]     # (N,)
            sdf = f["inputs/sdf"][:]     # (N, 128, 128)

            # Stack scalars into one tensor (N, 4)
            self.scalars = torch.tensor(
                list(zip(dbp, hr, phase, sbp)), dtype=torch.float32
            )

            # Convert sdf into (N,1,128,128)
            self.sdf = torch.tensor(sdf, dtype=torch.float32).unsqueeze(1)

            # Load labels
            u = torch.tensor(f["labels/u"][:], dtype=torch.float32).unsqueeze(1)
            v = torch.tensor(f["labels/v"][:], dtype=torch.float32).unsqueeze(1)
            p = torch.tensor(f["labels/p"][:], dtype=torch.float32).unsqueeze(1)

            # Concatenate into (N,3,128,128)
            self.y = torch.cat([u, v, p], dim=1)

        self.N = len(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # For now, we only return sdf maps as input
        # If you want scalars too, we can concatenate them later
        x = self.sdf[idx]   # (1,128,128)
        y = self.y[idx]     # (3,128,128)
        return x, y


def get_mcwilliams_loader(batch_size=2, val_split=0.2, data_root="data"):
    """
    Returns DataLoaders for train and validation splits.
    Also returns in_ch, out_ch, H, W for model initialization.
    """
    dataset = McWilliamsDataset(data_root=data_root, split="train")

    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Dimensions
    sample_x, sample_y = dataset[0]
    in_ch = sample_x.shape[0]   # should be 1 (sdf channel)
    out_ch = sample_y.shape[0]  # should be 3 (u,v,p)
    H, W = sample_x.shape[-2], sample_x.shape[-1]

    return train_loader, val_loader, in_ch, out_ch, H, W

