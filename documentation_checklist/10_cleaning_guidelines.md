

Cleaning should begin only after dataset identity and distribution have been verified. The first cleaning stage can then address issues discovered during EDA, such as unreadable files, accidental duplicates, corrupted images or clearly invalid samples.

## Cleaning Guidelines

- Do not resize images merely to make them uniform before recording their original dimensions.
    
- Do not remove minority-class samples simply because the class is small.
    
- Do not perform augmentation before establishing the raw class distribution.
    
- Keep a record of every removed, corrected or transformed sample.
    
- Preserve the original dataset separately from the cleaned or preprocessed copy.
    

## Current Status

These are the project's cleaning guidelines. Cleaning decisions should be documented after the raw image inspection and EDA findings have been reviewed.