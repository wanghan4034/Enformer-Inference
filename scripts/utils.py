import os
import numpy as np
from src.model import Enformer


# 需要的输出通道索引（保持固定顺序）
TRACK_IDXS = {
    'CHIP:H3K4me1:H1-hESC': 2291,
    'CHIP:H3K4me3:H1-hESC': 1509,
    'CHIP:H3K9me3:H1-hESC': 2653,
    'CHIP:H3K27me3:H1-hESC': 2034,
    'CHIP:H3K36me3:H1-hESC': 2911,
    'CHIP:H3K27ac:H1-hESC': 732,
    'CHIP:H3K9ac:H1-hESC': 4368,
}
TRACK_NAMES = list(TRACK_IDXS.keys())
TRACK_COLS  = np.array(list(TRACK_IDXS.values()), dtype=int)

# Enformer 的时域 & 目标分箱
WINDOW_BP   = 40_000         # 取中心 ±20kb
BIN_SIZE_BP = 500


def run_batch_predict(model: Enformer, inputs: np.ndarray):
    """对 (B, L, 4) 的输入做一次预测，返回 human 预测 (B, T, C)。"""
    outs = model.predict_on_batch(inputs)   # {'human': (B,T,C), ...}
    return outs['human']



def expand_to_genome(predictions: np.ndarray, base_res_bp: int = 128, 
                     is_clip: bool = True, window_bp: int = 40_000) -> np.ndarray:
    """
    将 Enformer 输出的 bin-level 预测展开到碱基分辨率，
    并可选地裁剪到中心 40kbp。

    参数：
        predictions (np.ndarray): (B, T, C) 的预测结果，
                                  其中 T=896，C=track数
        base_res_bp (int): 每个bin对应的碱基数，Enformer固定为128bp
        is_clip (bool): 是否裁剪到中心 window_bp（默认 False）
        window_bp (int): 裁剪窗口大小，默认 40kbp

    返回：
        np.ndarray: 
            如果 is_clip=False -> (B, T * base_res_bp, C)，展开到碱基分辨率
            如果 is_clip=True  -> (B, window_bp, C)，只保留中心 window_bp
    """
    # 展开到碱基分辨率
    expanded = np.repeat(predictions, base_res_bp, axis=1)  # (B,114688, C)

    if not is_clip:
        return expanded

    # 中心裁剪
    L = expanded.shape[1]
    half = window_bp // 2
    center = L // 2
    start = center - half
    end   = center + half
    return expanded[:,start:end, :]   # (B, window_bp, C)



def save_sample(gene_id: str,
                prediction: np.ndarray,
                tss: int = 0,
                chrom: str = '-',
                strand: str = '-',
                save_npy_dir: str | None = None):
    """
    从完整 (896, C) 输出，经“中心 40kb → 500bp/bin 重叠加权”得到 (80, 7)，按需保存。
    """
    # binned_dict = bin_center_40kb_from_full_896(prediction_2d, TRACK_COLS)  # {name: (80,)}
    binned_dict = {track_name: prediction[...,track_idx].astype(np.float32) for track_name, track_idx in TRACK_IDXS.items()}
    mat = np.stack([binned_dict[name] for name in TRACK_NAMES], axis=1)     # (80, 7) float32

    record = {
        "gene_id": gene_id,
        "chrom": chrom,
        "tss": int(tss),
        "strand": strand,
        "bins_bp": BIN_SIZE_BP,
        "window_bp": WINDOW_BP,
        "track_order": TRACK_NAMES,
    }

    os.makedirs(save_npy_dir, exist_ok=True)
    npy_path = os.path.join(save_npy_dir, f"{gene_id}.npy")
    np.save(npy_path, mat.astype(np.float32), allow_pickle=False)
    record["npy_path"] = npy_path
    return record
