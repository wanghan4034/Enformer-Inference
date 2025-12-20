import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EnformerDataset(Dataset):
    """
    Dataset for promoter-centered histone modification signals saved as .npy files.
    Each .npy file has shape (40000, 7), i.e., 40kbp window × 7 tracks.
    """

    def __init__(self, meta_csv, npy_dir, binsizes=[2000, 500, 100], w_prom=40000):
        """
        Args:
            meta_csv (str): Path to CSV with at least `gene_id, chrom, tss, strand, npy_path`.
            npy_dir  (str): Directory where npy files are stored.
            binsizes (list[int]): Bin sizes to produce promoter features.
            w_prom   (int): Promoter window size (default 40kbp).
        """
        self.meta = pd.read_csv(meta_csv)
        self.npy_dir = npy_dir
        self.binsizes = binsizes
        self.w_prom = w_prom

    def __len__(self):
        return len(self.meta)

    def _bin_and_pad(self, x, bin_size, max_n_bins):
        """
        Bin signal matrix by averaging values over `bin_size` bases,
        then pad to fixed `max_n_bins`.
        Returns binned tensor (n_feats × max_n_bins), plus padding indices.
        """
        L, C = x.shape  # L=40000, C=7
        n_bins = min(int(np.ceil(L / bin_size)), max_n_bins)

        # --- binning ---
        x_binned = []
        for i in range(n_bins):
            seg = x[i * bin_size : (i + 1) * bin_size]   # (bin_size, C)
            b = seg.mean(axis=0)                         # (C,)
            x_binned.append(b)
        x_binned = np.stack(x_binned, axis=0)            # (n_bins, C)

        # --- padding ---
        left_pad = int(np.ceil((max_n_bins - n_bins) / 2))
        right_pad = int(np.floor((max_n_bins - n_bins) / 2))
        x_binned = np.pad(
            x_binned,
            ((left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )  # (max_n_bins, C)

        return torch.tensor(x_binned, dtype=torch.float32), left_pad, n_bins, right_pad

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        gene_id = row["gene_id"]
        npy_path = os.path.join(self.npy_dir, os.path.basename(row["npy_path"]))
        x = np.load(npy_path)  # (40000, 7)

        item = {"gene_id": gene_id, "promoter_feats": {}, "promoter_pad_masks": {}}

        for binsize in self.binsizes:
            max_n_bins = self.w_prom // binsize
            x_binned, left_pad, n_bins, right_pad = self._bin_and_pad(x, binsize, max_n_bins)

            # (1, max_n_bins, n_feats)
            x_binned = x_binned.unsqueeze(0)

            # --- mask logic ---
            mask_p = torch.ones([1, max_n_bins, max_n_bins], dtype=torch.bool)
            mask_p[
                0,
                left_pad : left_pad + n_bins,
                left_pad : left_pad + n_bins,
            ] = 0
            mask_p = mask_p.unsqueeze(0)  # (1,1,max_n_bins,max_n_bins)

            item["promoter_feats"][binsize] = x_binned
            item["promoter_pad_masks"][binsize] = mask_p


        return item

if __name__ == "__main__":
    meta_csv = "tss_40kb_bins500bp_index.csv"
    npy_dir = "./binned_npys"

    dataset = EnformerDataset(meta_csv, npy_dir, binsizes=[2000, 500, 100])
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    batch = next(iter(loader))
    for binsize in [2000, 500, 100]:
        print(f"{binsize}bp promoter_feats:", batch["promoter_feats"][binsize].shape)
        print(f"{binsize}bp promoter_pad_masks:", batch["promoter_pad_masks"][binsize].shape)
