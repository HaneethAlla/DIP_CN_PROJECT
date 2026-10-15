# Local Dataset Verification

The team's local analysis covered eight unique Parquet files across the analyzed official splits.

## Verification Results

- Total rows across the eight analyzed files: 55,002.
    
- Number of WBC classes: 13.
    
- Class labels and counts were extracted programmatically.
    
- A descending class-distribution graph with count labels was generated.
    

## Local Class Counts

|Class|Local Count|
|---|---|
|SNE|25,532|
|LY|15,100|
|MO|5,194|
|BL|3,568|
|EO|1,640|
|BA|857|
|MY|792|
|VLY|711|
|MMY|650|
|BNE|648|
|PMY|192|
|PC|102|
|PLY|16|

## Verification Note

These counts describe the analyzed local Parquet files. They must not be treated as directly equivalent to the reference paper's Figure 1 training distribution of 38,535 images.

The local all-split distribution and the reference training distribution are retained separately for accurate comparison.