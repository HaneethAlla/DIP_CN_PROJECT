

Exploratory data analysis (EDA) begins with the raw dataset so that its original characteristics are documented before cleaning or preprocessing.

## Findings

The reference training distribution is strongly long-tailed. The local analysis of eight unique Parquet files also shows an imbalanced distribution across the 13 WBC classes.

The local class counts have been extracted programmatically, and a descending class-distribution graph with count labels has been generated.

## Comparison and Interpretation

- The reference paper's Figure 1 contains 38,535 training samples.
    
- The analyzed local Parquet files contain 55,002 rows across the analyzed splits.
    
- The two totals cover different scopes and must be interpreted separately.
    
- SNE is the largest class, while PLY is the smallest in both recorded distributions.
    

Class imbalance must be considered during later model training and evaluation. Macro-F1 is the official WBCBench evaluation metric.