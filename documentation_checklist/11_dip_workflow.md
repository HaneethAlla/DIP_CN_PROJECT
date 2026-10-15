

## Planned Workflow

1. Acquire and load the WBCBench 2026 dataset.
    
2. Verify the 13 class labels and class counts against the reference paper.
    
3. Perform EDA on the raw images.
    
4. Apply documented cleaning after EDA.
    
5. Standardize image dimensions during preprocessing if required by the selected classifier.
    
6. Apply suitable normalization and, where justified, augmentation.
    
7. Train a 13-class WBC image classifier.
    
8. Evaluate using class-wise metrics and macro-F1 in addition to overall accuracy.
    
9. Analyze confusion between visually similar WBC classes.
    

## Segmentation and Image-Quality Analysis

Cellpose segmentation and image-quality analysis are additional project activities being explored. Their results should be documented separately from the proposed classification workflow.

## Status

Dataset verification and class-count extraction are recorded as completed in the existing documentation. Cellpose processing has been attempted on a balanced image sample, but the final segmentation outputs must be verified before the segmentation task is marked complete.

The remaining workflow steps should be marked completed only when supported by actual results.