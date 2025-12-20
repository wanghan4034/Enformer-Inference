import os
# —— Set before importing TensorFlow to avoid allocating all GPU memory at once ——
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import kipoiseq
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.core.model import Enformer
from src.core.utils import FastaStringExtractor, one_hot_encode
from src.core.constants import SEQUENCE_LENGTH
from scripts.utils import expand_to_genome, save_sample

# ================= Basic configuration =================
model_path   = 'checkpoints/tensorflow_v1'
fasta_file   = './extra/hg19.fa'
genes_bed    = './data/genes_E003.bed'   # must include: chrom, start, end, strand, gene_id
batch_size   = 2
window_size  = 40_000                         # only used to extract sequence around the TSS; the model still sees SEQUENCE_LENGTH
save_npy_dir = f'./outputs/E003/regions'                 # set to None if you do not want to save npy



# ================= Model and FASTA =================
model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)

# ================= Utility functions =================
def build_input_for_tss(chrom: str, tss: int, strand: str):
    """Extract a window around the TSS, then resize to SEQUENCE_LENGTH; return one-hot and interval."""
    half = window_size // 2
    interval = kipoiseq.Interval(chrom, tss - half, tss + half, strand=strand)
    target_interval = interval.resize(SEQUENCE_LENGTH)
    seq = fasta_extractor.extract(target_interval)
    return one_hot_encode(seq), target_interval

def run_batch_predict(model: Enformer, inputs: np.ndarray):
    """Run a single prediction for inputs of shape (B, L, 4); return human predictions (B, T, C)."""
    outs = model.predict_on_batch(inputs)   # {'human': (B,T,C), ...}
    return outs['human']



# ================ Main workflow (batch inference + binning + saving) =================
def main():
    genes = pd.read_csv(genes_bed, sep='\t')
    genomic_tracks = {}  # gene_id -> record
    batch_inputs = []    # (L, 4)
    batch_meta   = []    # (gene_id, chrom, tss, strand)

    pbar = tqdm(genes.itertuples(index=False), total=len(genes))
    for row in pbar:
        strand = row.strand
        start  = int(row.start)
        end    = int(row.end)
        tss    = start if strand == '+' else end

        one_hot, _ = build_input_for_tss(row.chrom, tss, strand)
        batch_inputs.append(one_hot)
        batch_meta.append((row.gene_id, row.chrom, tss, strand))

        if len(batch_inputs) == batch_size:
            batch_np = np.asarray(batch_inputs)                # (B, L, 4)
            preds_btC = run_batch_predict(model, batch_np)            # (B, 896, C)
            expended = expand_to_genome(preds_btC)
            for b, (gene_id, chrom, tss_b, strand_b) in enumerate(batch_meta):
                sample_prediction = expended[b]
                record = save_sample(gene_id, sample_prediction, tss_b, chrom, strand_b, save_npy_dir)
                genomic_tracks[gene_id] = record
            batch_inputs.clear()
            batch_meta.clear()

    # tail batch
    if batch_inputs:
        batch_np = np.asarray(batch_inputs)
        preds_btC = run_batch_predict(model, batch_np)
        expended = expand_to_genome(preds_btC)
        for b, (gene_id, chrom, tss_b, strand_b) in enumerate(batch_meta):
            sample_prediction = expended[b]
            record = save_sample(gene_id, sample_prediction, tss_b, chrom, strand_b, save_npy_dir)
            genomic_tracks[gene_id] = record
        batch_inputs.clear()
        batch_meta.clear()


    # Save an index table for convenient downstream loading
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
    pd.DataFrame(index_rows).to_csv(f'./outputs/E003/meta_data.csv', index=False)
    print("Saved index:", 'meta_data.csv')

    return genomic_tracks

if __name__ == "__main__":
    _ = main()