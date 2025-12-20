import numpy as np
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import matplotlib.pyplot as plt
import seaborn as sns


SEQUENCE_LENGTH = 393216


# @title `variant_centered_sequences`

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)

  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)



def map_tracks_to_genome(tracks, interval):
    """
    将多个组蛋白修饰的track向量映射到指定的基因组 interval 上。

    参数：
        tracks (dict): 形如 {track_name: 1D array-like values} 的字典
        interval (Interval): 拥有 start 和 end 属性
    返回：
        dict: {track_name: 1D ndarray of values for each position in interval}
    """
    start = interval.start
    end = interval.end
    positions = np.arange(start, end)
    mapped = {}

    for name, y in tracks.items():
        y = np.array(y, dtype=np.float32)
        bins = np.linspace(start, end, num=len(y) + 1, dtype=int)

        bin_indices = np.searchsorted(bins, positions, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(y) - 1)

        mapped[name] = y[bin_indices]

    return mapped



def bin_40kbp_matrix(matrix: np.ndarray, bin_size_bp: int = 500) -> np.ndarray:
    """
    将 40kbp × tracks 的矩阵，按 bin_size_bp 分箱平均。
    
    参数：
        matrix: np.ndarray, shape=(40000, C)，40kbp 范围的预测结果
        bin_size_bp: int, 每个 bin 的碱基长度，默认 500bp
    
    返回：
        np.ndarray, shape=(N_bins, C)，N_bins = 40000/bin_size_bp
    """
    length, C = matrix.shape
    assert length == 40000, f"输入长度应为40kbp，这里是 {length}"
    
    n_bins = length // bin_size_bp  # 80
    binned = matrix.reshape(n_bins, bin_size_bp, C).mean(axis=1)
    return binned



# 7 个 histone track 的名字（保持保存时的顺序）


def plot_tracks(tracks, ylim=None, is_binned=True):
    """
    绘制多个组蛋白修饰的轨迹。

    参数：
        tracks (np.ndarray): 形状为 (N_bins, N_tracks) 的矩阵
        track_names (list): 轨迹名称列表
    """
    track_names = [
    'H3K4me1',
    'H3K4me3',
    'H3K9me3',
    'H3K27me3',
    'H3K36me3',
    'H3K27ac',
    'H3K9ac'
    ]
    if is_binned:
        tracks = bin_40kbp_matrix(tracks)
    # 横轴坐标：-20kb 到 +20kb
    x = (np.arange(tracks.shape[0]) - (tracks.shape[0] // 2 + 1)) * 500


    plt.figure(figsize=(8/2.54, 3.5/2.54), dpi=300)
    for i, name in enumerate(track_names):
        plt.plot(x, tracks[:, i], label=name, lw=0.5)

    plt.axvline(0, color='k', linestyle='--', lw=0.5, label="TSS")
    if ylim:
       plt.ylim(top=ylim)
    plt.xlabel("Position relative to TSS (bp)")
    plt.ylabel("Signal intensity")
    plt.title("Histone modification profiles around TSS")
    plt.legend(
        bbox_to_anchor=(1.1, 1.05),  # 放在右上角外面
        loc='upper left',
        fontsize=5

    )
    plt.tight_layout()
    plt.show()
