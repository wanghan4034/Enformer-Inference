# **DeepBurst Enformer Inference Utilities**

This submodule provides lightweight, script-based utilities for generating sequence variants and running **Enformer** inference to obtain sequence-predicted histone-mark tracks that can be used as inputs to [**DeepBurst**](https://github.com/wanghan4034/DeepBurst/). The workflow uses **DeepMind’s official TensorFlow Enformer implementation** and the **official pre-trained checkpoints** referenced by DeepMind.

## **Overview**

This submodule supports two primary use cases:

1. **TSS-centred Enformer inference at scale**

   Given a BED file of genes/regions, extract TSS-centred sequences from a reference genome and run batched Enformer inference. Predictions are saved to disk along with a metadata index for downstream loading.

2. **Sequence perturbation experiments via centre-segment replacement**

   For a single locus, generate many synthetic sequence variants by replacing a central segment of the reference sequence with random DNA, then run Enformer inference on all variants to obtain corresponding *in silico* histone-mark profiles.

The outputs are designed to be directly consumable by DeepBurst for downstream prediction and analysis.



## **What this submodule produces**

- Per-region / per-variant Enformer predictions saved as arrays (e.g., .npy, depending on configuration).
- A meta_data.csv index that records sequence/region identifiers and corresponding output locations for downstream loading and analysis.



## **Design constraints**

- **TensorFlow-only workflow**

  This submodule targets DeepMind’s official TensorFlow Enformer pipeline (it is not intended for PyTorch re-implementations).

- **Official checkpoints**

  The model must be loaded from DeepMind’s official pre-trained Enformer checkpoints (or the official hosting referenced by DeepMind).

- **Reference FASTA required**

  Scripts assume a local reference genome FASTA (e.g., hg19.fa) for sequence extraction.



## **Compute environment**

Inference runs in this project were executed on an **NVIDIA A800 GPU (80 GB VRAM)**.

To avoid pre-allocating all GPU memory at startup, the scripts enable TensorFlow GPU memory growth (environment variables and TensorFlow settings as appropriate for your system).

## **Installation**

### **Python dependencies**

This submodule is tested with **Python 3.10**. Install dependencies from requirements.txt:

```
conda create -n enformer-inference python=3.10 -y
conda activate enformer-inference
pip install -r requirements.txt
```



## **Model checkpoints (official DeepMind Enformer)**

This submodule expects the **official DeepMind TensorFlow Enformer checkpoints**.

Set the checkpoint location in the scripts (example):

```
MODEL_PATH = "checkpoints/tensorflow_v1"
```

Ensure the directory structure matches what the Enformer loader in this submodule expects.

## **Quickstart**

### **A) TSS-centred inference for a BED of genes**

```
python enformer_predictor.py
```

Outputs (example):

- ./outputs/<eid>_enformer_prediction/regions/
- ./outputs/<eid>_enformer_prediction/meta_data.csv



### **B) Random centre-segment replacement experiment**

1. Generate randomized FASTA variants:

```
python generate_center_dna.py
```



1. Run Enformer predictions on the generated FASTA:

```
python prediction_from_random_seq.py
```

Outputs (example):

- ./outputs/<gene_id>_random_center_replaced_<N>/regions/
- ./outputs/<gene_id>_random_center_replaced_<N>/meta_data.csv

## **Attribution and licensing**

This submodule builds on DeepMind’s official Enformer reference implementation and usage workflow:

```
https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb
```

If you publish results using this submodule, please cite Enformer appropriately and comply with DeepMind’s licensing and usage requirements for the official code and checkpoints.