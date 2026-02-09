# A multi-scale and multi-perspective 5' UTR translation efficiency prediction model

## Abstract

### Background
The 5′untranslated region (5′UTR) is a critical cis-regulatory element governing eukaryotic mRNA translation initiation. While translation efficiency depends on both sequence composition and RNA secondary structure, most current computational models rely primarily on sequence features, neglecting explicit structural regulatory information.

### Results
We introduce a multi-scale, multi-view deep learning framework that jointly learns local and global features from 5′UTR sequences and their predicted secondary structures. Utilizing a bidirectional long short-term memory (BiLSTM) network with an attention mechanism, this model integrates local and global data to predict translation efficiency. It consistently outperforms existing approaches across diverse synthetic and endogenous benchmark datasets. Ablation analyses demonstrate that the explicit incorporation of RNA structural features significantly enhances predictive performance. Attention-based interpretability analyses further demonstrate that the model effectively captures a broad range of translational regulatory signals, including upstream AUGs (uAUGs), local stability, and structural accessibility near the main start codon. Demonstrating practical utility, we rationally designed 14 high-efficiency 5′UTRs using this framework; experimental validation showed these designs achieved up to a 4.75-fold increase in protein expression compared to controls.

### Conclusions
Our findings highlight the necessity of integrating RNA secondary structure into predictive models. As data resources continue to expand and modeling techniques advance, such integrative frameworks are expected to further deepen our mechanistic understanding of 5′UTR-mediated translational control and provides a robust foundation for variant interpretation and the synthetic design of optimized regulatory elements.


## Requirements

### Python Version
- Python >= 3.8

### Core Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ~1.26.4 | Numerical computing |
| pandas | ~2.2.2 | Data manipulation |
| scikit-learn | ~1.5.1 | Machine learning utilities (train_test_split, scalers) |
| tensorflow | ~2.17.0 | Deep learning framework (BiLSTM, Attention) |
| biopython | ~1.84 | Sequence I/O (SeqIO for FASTA files) |
| joblib | ~1.4.2 | Parallel proces
| viennarna | ~2.4.7 | RNAFold


## Project Structure

The project is organized into the following directories:

```text
.
├── GA/                 # Implementation of Genetic Algorithms for 5' UTR optimization
├── data/               # Datasets used for training and testing
│   ├── endogenous/     # Example datasets from endogenous sources
│   └── synthetic/      # Example datasets from synthetic sources
├── modelling/          # Source code for model construction and training
│   ├── endogenous/     # Scripts for building the endogenous prediction model
│   └── synthetic/      # Scripts for building the synthetic prediction model
└── models/             # Saved model files and pre-processing parameters
    ├── GA/             # Stored parameters required for data normalization (e.g., scalers) (pkl format)
    ├── endogenous/     # Pre-trained endogenous models (H5 format)
    └── synthetic/      # Pre-trained synthetic models (H5 format)
