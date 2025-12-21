# **DeepBurst Enformer Inference Utilities (TensorFlow / DeepMind Official)**

This submodule provides lightweight, script-based utilities to (1) generate random DNA variants by replacing the central segment of a target sequence, and (2) run **Enformer** inference to produce histone-mark tracks as inputs to **DeepBurst**, using **DeepMind’s official TensorFlow implementation** and the **official pre-trained checkpoints** from the DeepMind Enformer repository.

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

### **1)** enformer_predictor.py

Runs **batched Enformer inference** for a list of genes provided in a BED file. For each gene, it identifies the **TSS**, extracts a **TSS-centered sequence window**, resizes it to SEQUENCE_LENGTH, performs prediction, and saves the outputs plus an index table.

### **2)** generate_center_dna.py

Creates a **random-sequence perturbation dataset** for a single target locus. It extracts the reference sequence and **replaces the center segment** with random DNA for many samples, writing all sequences to a **gzipped FASTA** file.

### **3)** prediction_from_random_seq.py

Runs **batched Enformer inference** on the random FASTA generated above. It reads the gzipped FASTA, **pads each sequence to SEQUENCE_LENGTH**, performs prediction, saves per-sequence outputs, and writes a corresponding meta_data.csv index for downstream analysis.

## **Installation**

### **Python dependencies**

An environment snapshot is provided in requirements.txt. This submodule is tested with **Python 3.10**.

```
conda create -n enformer-inference python=3.10 -y
conda activate enformer-inference
pip install -r requirements.txt
```

## **Model checkpoints (official DeepMind Enformer)**

This submodule expects the **DeepMind official TensorFlow Enformer checkpoints** as provided by the official DeepMind Enformer repository and its documentation.

Configure the checkpoint location in scripts via:

- MODEL_PATH = "checkpoints/tensorflow_v1" 

Ensure the directory structure matches what your src.core.model.Enformer loader expects.

## **Quickstart**

### **A) TSS-centered inference for a BED of genes**

```
python enformer_predictor.py
```

Outputs:

- ./outputs/\<eid>_enformer_prediction/regions/
- ./outputs/\<eid>_enformer_prediction/meta_data.csv

### **B) Random-center replacement experiment**

1. Generate randomized FASTA variants:

```
python generate_center_dna.py
```

2. Run Enformer predictions on the generated FASTA:

```
python prediction_from_random_seq.py
```

Outputs:

- ./outputs/<gene_id>_random_center_replaced_100k/regions/
- ./outputs/<gene_id>_random_center_replaced_100k/meta_data.csv

## **Attribution**

The Enformer inference flow is based on DeepMind’s official TensorFlow reference implementation and is adapted from the official usage notebook:

https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb

If you publish results using this submodule, please cite Enformer appropriately and follow DeepMind’s license/usage requirements for the official code and checkpoints.