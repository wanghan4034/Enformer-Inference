import random
import gzip
from src.utils import FastaStringExtractor
import kipoiseq

seed = 123
random.seed(seed)
# ================= Basic configuration =================
fasta_file = './extra/hg19.fa'
fasta_extractor = FastaStringExtractor(fasta_file)

# Select the TSS region of a gene for center replacement
gene_id = 'ENSG00000155660'  
chrom = 'chr7'
start = 148705597
end = 148705597
strand = '-'

# gene_id = 'ENSG00000143340'
# chrom = 'chr1'
# start = 179692424
# end = 179732424
# strand = '+'
interval = kipoiseq.Interval(chrom, start, end, strand=strand)
target_seq = fasta_extractor.extract(interval)

replaced_length = 2000
num_samples = 100   # change this to 100k
output_path = f"{gene_id}_random_center_replaced_100k.fa.gz"

# ================= Utility functions =================
def random_dna(length: int, alphabet="ACGT") -> str:
    """Generate a random DNA string of given length."""
    return "".join(random.choices(alphabet, k=length))

def random_replace_center_seq(target_seq: str, replaced_length: int) -> str:
    """Replace the center segment of target_seq with a random DNA string."""
    total_length = len(target_seq)
    random_seq = random_dna(length=replaced_length)
    left = (total_length - replaced_length) // 2
    right = total_length - left - replaced_length
    return target_seq[:left] + random_seq + target_seq[-right:]

def make_ids(num_samples: int, prefix: str = "ENSGRND"):
    """Generate IDs like ENSGRND00000001."""
    width = max(8, len(str(num_samples)))
    for i in range(1, num_samples + 1):
        yield f"{prefix}{i:0{width}d}"

def save_fasta_gz(sequences, ids, output_path, line_width=80):
    """Save sequences to a gzipped FASTA file."""
    with gzip.open(output_path, "wt") as f:
        for seq_id, seq in zip(ids, sequences):
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + "\n")

# ================= Main workflow =================
if __name__ == "__main__":
    sequences = [random_replace_center_seq(target_seq, replaced_length) for _ in range(num_samples)]
    ids = list(make_ids(num_samples))
    save_fasta_gz(sequences, ids, output_path)
    print(f"Saved {num_samples} sequences to {output_path}")