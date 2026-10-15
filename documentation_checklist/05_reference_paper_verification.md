

The reference paper used for distribution verification is _Synergizing Deep Learning and Biological Heuristics for Extreme Long-Tail White Blood Cell Classification_.

The paper states that WBCBench 2026 contains 55,012 peripheral blood smear images across 13 fine-grained WBC classes.

Figure 1 of the paper shows the training-set class distribution. The 13 values shown in that figure sum to 38,535. Therefore, these values represent the reported training-set distribution, not the total number of images in the complete dataset.

## Verification Finding

The largest class in Figure 1 is SNE, with 17,354 images. The smallest is PLY, with 14 images.

The ratio between these counts is approximately 1,239.6:1, consistent with the paper's description of an approximately 1,200-fold long-tail imbalance.

The reference-paper distribution must be kept separate from local counts because they cover different dataset scopes.