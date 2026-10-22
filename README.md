# TF-CORE: Training-Free In-Context Medical Image Segmentation via Prototype Refinement and Self-Correction

## Abstract

Training-free in-context segmentation (ICS) enables medical image segmentation using one annotated reference image without parameter updates, reducing reliance on dense pixel-level annotations and task-specific training. Using a frozen visual model, it transfers reference semantics to an unlabeled target through reference-target feature matching. However, existing ICS methods face three limitations: foreground heterogeneity and foreground-like backgrounds lead to ambiguous prototype matching; mask refinement lacks targeted propagation and explicit correction guidance; and pixel-wise semantic predictions are insufficiently constrained by intrinsic target structure. We propose TF-CORE, a training-free self-correction framework that combines discriminative feature matching, reference-guided error correction, and structural refinement within the feature space of a single frozen encoder. First, Instance-level Dual-check Prototypes (IDP) represents diverse foreground and hard-background patterns with multiple prototypes, while Evidence Accumulation Matching (EAM) aggregates their evidence to produce a discriminative initial prediction. Second, Maximum Pooling Diffusion (MPD) and Bidirectional Verification and Correction (BVC) expand foreground responses through target feature affinities and iteratively correct errors through verification against the annotated reference. Third, Structure-guided Competitive Refinement (SCR) combines semantic correspondence with intrinsic target structure to resolve inconsistencies between semantic predictions and target-image structure. Across four datasets covering different imaging modalities and segmentation targets, TF-CORE achieves the best overall performance among the evaluated training-free ICS methods, demonstrating the effectiveness of our proposed method for training-free medical image segmentation with limited annotations.

## Datasets

- Chest X-ray (Montgomery County and Shenzhen): [Download](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- Kvasir-SEG: [Download](https://datasets.simula.no/kvasir-seg/)
- BUSI: [Download](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- GlaS: [Download](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest-backup/download/)

## Updates

This repository currently provides the framework. The code and documentation will be updated progressively.
