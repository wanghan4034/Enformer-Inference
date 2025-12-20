# **DeepBurst Enformer Inference Utilities (TensorFlow / DeepMind Official)**

This submodule provides lightweight, script-based utilities to (1) generate random DNA variants by replacing the center of a target sequence, and (2) run **Enformer** inference using **DeepMind’s official TensorFlow implementation** and **official pre-trained checkpoints** from the DeepMind Enformer repository.

**Important implementation note.** The core Enformer inference logic in this submodule is adapted from DeepMind’s official Colab notebook (with project-specific modifications for batching, I/O, and genomic-coordinate expansion):

https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb

## **What this submodule does**

- **TSS-centered inference for many genes** (BED → per-gene Enformer predictions → saved arrays + metadata index).
- **Random-center replacement experiments** for a specific locus (reference sequence → many randomized center variants → Enformer predictions for each variant).
- Produces a **CSV index** (meta_data.csv) for downstream loading and analysis, plus optional .npy prediction artifacts.



## **Design constraints (read this first)**

- **TensorFlow only**: This submodule is intended for DeepMind’s **official TensorFlow Enformer** workflow (not PyTorch ports).
- **Official pre-trained model**: The model should be loaded from the **official DeepMind Enformer checkpoint(s)** distributed via the official code repository (or the official hosting referenced there).
- **Reference FASTA required**: Scripts assume a local reference genome FASTA (e.g., hg19.fa) for sequence extraction.



## **Compute environment**

All inference runs in this project are executed on an **NVIDIA A800 GPU with 80 GB VRAM** (A800 80G).

The scripts enable TensorFlow GPU memory growth to avoid allocating all VRAM at startup:

- TF_FORCE_GPU_ALLOW_GROWTH=true
- tf.config.experimental.set_memory_growth(...)



## **Repository layout (scripts)**

### **1)** **enformer_predictor.py**

Runs **batch inference** for a gene list (BED). For each gene, it extracts a window around the **TSS**, resizes to Enformer input length (SEQUENCE_LENGTH), and performs prediction.

- Inputs:

  - genes_bed: must include chrom, start, end, strand, gene_id
  - Reference FASTA
  - Enformer checkpoint directory (TensorFlow)

- Outputs:

  - Per-gene saved predictions (typically .npy), path returned by save_sample
  - ./outputs/<dataset>/meta_data.csv index

  

### **2)** **generate_center_dna.py**

Generates many sequences by **replacing the center** of a target reference sequence with random DNA.

- Inputs:
  - Reference FASTA (e.g., ./extra/hg19.fa)
  - A target interval (typically TSS-centered; configurable)
  - replaced_length (e.g., 2000 bp)
  - num_samples
- Output:
  - gzipped FASTA: "{gene_id}_random_center_replaced_100k.fa.gz"

### **3)** **prediction_from_random_seq.py**

Reads the gzipped FASTA produced by generate_center_dna.py, **pads** each sequence to Enformer’s input length (SEQUENCE_LENGTH) by centering it, runs batch inference, and saves outputs.

- Inputs:
  - FA_GZ_PATH: "{gene_id}_random_center_replaced_100k.fa.gz"
  - Enformer checkpoint directory (TensorFlow)
- Outputs:
  - Per-sequence saved predictions (typically .npy)
  - ./outputs/{gene_id}_random_center_replaced_100k/meta_data.csv



## **Installation**

### **Python dependencies**

A raw environment snapshot is provided here: 

In most cases, the minimal runtime dependencies for these scripts include:

- tensorflow (GPU recommended)
- tensorflow-hub (depending on how the Enformer wrapper loads weights)
- numpy, pandas, tqdm
- biopython
- kipoiseq
- plus project-local modules under src/ and utility functions under scripts/ / src.scripts/

------





## **Model checkpoints (official DeepMind Enformer)**



This submodule expects the **DeepMind official TensorFlow Enformer checkpoints** as provided by the official DeepMind Enformer repository and its documentation.

Configure the checkpoint location in scripts via:

- MODEL_PATH = "checkpoints/tensorflow_v1" 

Ensure the directory structure matches what your src.model.Enformer loader expects.



## **Input data requirements**

### **Reference genome FASTA**

Example:

- ./extra/hg19.fa

### **Gene BED (for** 

### **enformer_predictor.py**

### **)**

Must contain columns:

- chrom, start, end, strand, gene_id

TSS is computed as:

- tss = start if strand == '+'
- tss = end if strand == '-'



## **Outputs**

All inference scripts create:

1. A directory of saved prediction artifacts (commonly .npy)
2. A meta_data.csv index with fields:

- gene_id
- chrom
- tss
- strand
- npy_path
- bins_bp
- window_bp
- track_order (pipe-delimited)



## **Quickstart**

### **A) Random-center replacement experiment**

1. Generate randomized FASTA variants:

```
python generate_center_dna.py
```



1. Run Enformer predictions on the generated FASTA:

```
python prediction_from_random_seq.py
```

Outputs:

- ./outputs/<gene_id>_random_center_replaced_100k/regions/
- ./outputs/<gene_id>_random_center_replaced_100k/meta_data.csv



### **B) TSS-centered inference for a BED of genes**

```
python enformer_predictor.py
```

Outputs:

- ./outputs/<dataset>/regions/
- ./outputs/<dataset>/meta_data.csv

## **Attribution**

The Enformer inference flow is based on DeepMind’s official TensorFlow reference implementation and is adapted from the official usage notebook:

https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb

If you publish results using this submodule, please cite Enformer appropriately and follow DeepMind’s license/usage requirements for the official code and checkpoints.