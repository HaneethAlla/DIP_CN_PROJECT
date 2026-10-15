#

The following code loads the pristine configuration of WBCBench 2026, counts the class labels and calculates image resolutions and channel configurations.

```
from datasets import load_dataset
from collections import Counter
import numpy as np

REPO = "Xin-Tian/wbcbench2026"

ds = load_dataset(
    REPO,
    "pristine",
    split="phase2_train"
)

counts = Counter(ds["label"])

print("Total images:", len(ds))
print("\nClass counts:")

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

print("\nResolutions:")

resolution_counts = Counter()
channel_counts = Counter()

for row in ds:
    img = row["image"]
    resolution_counts[img.size] += 1
    channel_counts[len(img.getbands())] += 1

print(resolution_counts)
print("\nChannels:", channel_counts)
```

**Note:** This code analyzes the specified `phase2_train` split. Its results should not be assumed to reproduce the counts from all eight locally analyzed Parquet files.