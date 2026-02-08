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
| numpy | ~1.24.0 | Numerical computing |
| pandas | ~2.0.0 | Data manipulation |
| scikit-learn | ~1.3.0 | Machine learning utilities (train_test_split, scalers) |
| tensorflow | ~2.13.0 | Deep learning framework (BiLSTM, Attention) |
| biopython | ~1.81 | Sequence I/O (SeqIO for FASTA files) |
| joblib | ~1.3.0 | Parallel proces
