import os
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from Bio import SeqIO

from src.model import Enformer
from src.constants import SEQUENCE_LENGTH
from src.utils import one_hot_encode  #
from src.scripts.utils import run_batch_predict, expand_to_genome, save_sample

# ---------------- Config ----------------
gene_id = "ENSG00000155660"  # Example gene ID
FA_GZ_PATH   = f"{gene_id}_random_center_replaced_100k.fa.gz"                 # 输入文件
MODEL_PATH   = "checkpoints/tensorflow_v1"
BATCH_SIZE   = 8
READ_HEAD    = "human"                      # "human" or "mouse"

# ---------------- GPU 设置 ----------------
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] set_memory_growth failed on {gpu}: {e}")

# ============= Pad/Crop 到 Enformer 长度 =============
def to_enformer_length_center(seq_oh: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Pad/crop one-hot seq to Enformer input length (393,216 bp)."""
    L = seq_oh.shape[0]
    if L == target_len:
        return seq_oh
    out = np.zeros((target_len, 4), dtype=np.float32)
    assert L <= target_len, f"Input length {L} exceeds target {target_len}."
    start = (target_len - L) // 2
    out[start:start+L] = seq_oh
    return out

# ============= Batch Reader =============
import gzip
from Bio import SeqIO
import numpy as np
import gzip
from Bio import SeqIO
import numpy as np

class Reader:
    def __init__(self, path, batch_size) -> None:
        self.path = path
        self.batch_size = batch_size
        self._length = self._count_records()  # count total number of sequences

    def _count_records(self):
        """Count how many sequences are in the fasta file."""
        n = 0
        with gzip.open(self.path, "rt") as f:
            for _ in SeqIO.parse(f, "fasta"):
                n += 1
        return n

    def __len__(self):
        """
        Return the number of batches.
        If you prefer returning the number of sequences instead,
        just return self._length.
        """
        return (self._length + self.batch_size - 1) // self.batch_size

    def fasta_batches(self):
        """
        Yield batches of sequences from a gzipped fasta file.

        Yields:
            ids (list[str]): list of sequence IDs in the batch
            samples (np.ndarray): (B, L, 4) one-hot encoded sequences
        """
        ids, samples = [], []
        with gzip.open(self.path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq = str(record.seq).upper()
                sample = one_hot_encode(seq)
                sample = to_enformer_length_center(sample, SEQUENCE_LENGTH)
                ids.append(record.id)
                samples.append(sample)
                if len(samples) == self.batch_size:
                    yield ids, np.asarray(samples, dtype=np.float32)
                    ids, samples = [], []
        # Yield the last incomplete batch
        if samples:
            yield ids, np.asarray(samples, dtype=np.float32)

# ============= Main =============
def main():
    model = Enformer(MODEL_PATH)
    reader = Reader(FA_GZ_PATH, BATCH_SIZE)
    genomic_tracks = {}
    save_npy_dir = f'./outputs/{gene_id}_random_center_replaced_100k/regions'   
    for sample_ids, samples in tqdm(reader.fasta_batches(), desc="Predict", total=len(reader)):
        outs = run_batch_predict(model, samples)
        # preds = outs[READ_HEAD]   # (B, 896, C)
        expended = expand_to_genome(outs)
        for idx, sample_id in  enumerate(sample_ids):
            sample_prediction = expended[idx]
            record = save_sample(sample_id, sample_prediction, save_npy_dir=save_npy_dir)
            genomic_tracks[sample_id] = record

    index_rows = []
    for gid, rec in genomic_tracks.items():
        index_rows.append({
            "gene_id": gid,
            "chrom": rec["chrom"],
            "tss": rec["tss"],
            "strand": rec["strand"],
            "npy_path": rec["npy_path"],
            "bins_bp": rec["bins_bp"],
            "window_bp": rec["window_bp"],
            "track_order": "|".join(rec["track_order"]),
        })
    pd.DataFrame(index_rows).to_csv(f'outputs/{gene_id}_random_center_replaced_100k/meta_data.csv', index=False)
    print("Saved index:", 'meta_data.csv')

if __name__ == "__main__":
    main()