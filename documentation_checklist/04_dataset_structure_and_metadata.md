

- The Hugging Face release provides pristine and degraded configurations.
    
- Official splits include `phase1_train`, `phase2_train`, `phase2_eval` and `phase2_test`, as applicable to each configuration.
    
- The image field contains JPEG image bytes that can be decoded as PIL images using the Hugging Face Datasets library.
    
- The dataset card reports 493 unique patients and patient-level disjointness across official splits.
    
- The official ranking metric is macro-averaged F1 across all 13 classes.
    
- The degraded configuration contains pristine, mild, moderate and extreme severity levels.
    

## Available Metadata

The dataset records information such as:

- Image bytes
    
- Image ID
    
- Class label
    
- Official split
    
- Degradation severity
    
- Original image ID for degraded samples
    
- Patient hash
    

This metadata supports dataset inspection, class-distribution analysis and robustness evaluation.