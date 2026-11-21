# A Robust 13-Class White Blood Cell Classification System with Edge–Fog Offloading for Resource-Constrained Cancer Screening

**A. Haneeth** — Department of Artificial Intelligence and Data Science  
**Nivrutha Varshini** — Department of Computer Science and Engineering  
IIIT Raichur, India  
Guides: Dr. Mehbub Alam, Dr. Koushiki Roy

**Submission artifact — WBCBench 2026 / DIP–CN project**

**Paper** — [Final manuscript (PDF)](paper/PAPER.pdf)

**Live deployment:** https://dipcnproject-iznyisdqoq6lr7tjnyw9nu.streamlit.app/

---

## Main results

| Evaluation | Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| *phase2_eval* | Best single | 0.9294 | 0.7114 | 0.7144 | 0.9300 |
| *phase2_eval* | Top-3 TTA ensemble | 0.9405 | 0.7242 | 0.7325 | 0.9396 |
| *phase2_test* | Best single | 0.9474 | 0.7104 | 0.6811 | 0.9502 |
| *phase2_test* | Top-3 TTA ensemble | 0.9591 | 0.7115 | 0.7037 | 0.9599 |
| 5-fold CV | Training pool | 0.9490 ± 0.0037 | — | 0.8461 ± 0.0114 | — |

| Edge–fog result | Value |
|---|---:|
| Always-local accuracy / latency | 0.848 / 13.2 ms |
| Learned policy accuracy / latency | 0.870 / 30.2–30.4 ms |
| Always-fog accuracy / latency | 0.914 / 484–491 ms |
| Oracle accuracy / latency | 0.926 / 50.2–50.7 ms |
| Learned policy error-aware cost | 160.2–160.4 ms |
| Always-local error-aware cost | 165.2 ms |
| Real-TCP success and integrity | 20/20, 100% |

The routing objective is **error-aware cost = latency + 1,000 ms × 1[error]**, not raw latency. The learned policy captures about one-third of the accuracy gap between local and fog, adds about 17 ms over local, and avoids roughly 455 ms of fog latency on WiFi/5G.

---

## System overview

```text
Uploaded image ──┬──> Quality features ─────┐
                 └──> Edge INT8 prediction ─┼──> Learned policy
                      Network profile ──────┘         │
                                                       ├── LOCAL: use Edge prediction
                                                       └── OFFLOAD: run Fog ensemble over TCP
```

- **DIP model:** EfficientNet-B0, 13 WBC classes.
- **Edge model:** static INT8 ONNX, CPU inference, no PyTorch needed.
- **Fog model:** three-checkpoint EfficientNet ensemble with eight-view TTA.
- **Routing:** a depth-4 decision tree selects local Edge inference or Fog offload using image quality, Edge confidence, and network-profile features.
- **Transport:** framed TCP with SHA-256 image integrity, ACK/retransmit, and a threaded Fog server.

---

## Submission contents

| Path | Contents |
|---|---|
| `DIP/training/app.py` | Streamlit demonstration entry point |
| `DIP/training/src/` | Quality features, data, model, training, and evaluation modules |
| `DIP/training/checkpoints/` | Three runtime Fog ensemble checkpoints |
| `DIP/training/data/manifests/` | Image-level manifests and fold assignments |
| `DIP/training/results/` | Final training, validation, test, ensemble, and 5-fold metrics |
| `DIP/cellpose/` | Cellpose negative-result experiment and evidence |
| `CN/src/` | TCP transport, Fog backend, Edge predictor, and routing policy |
| `CN/results/cn_edge_fog/` | Policy matrix, TCP validation, selected figures, and policy tree |
| `CN/tests/` | Transport, backend, policy, and integration tests |
| `model_share/deploy/` | Quantized INT8 ONNX model and deployment wrapper |
| `model_share/results/` | Edge/Fog evaluation metrics and latency records |
| `feature_analysis/` | Quality-feature extraction code, manifests, and severity summaries |

Dataset images are **not** included. `phase2_test` is reserved for final evaluation and was not used for tuning. The official `patient_hash` metadata was not downloaded, so patient-level leakage remains unverified; the five-fold cross-validation is image-level.

---

## Reproducing the results

### Environment

Use Python 3.10 on Linux. CPU inference supports the demo and evaluation. A compatible CUDA installation is optional for GPU training. Internet access is needed to install dependencies, obtain WBCBench through the organizers, and download pretrained EfficientNet weights if retraining. The shipped Edge ONNX model, routing policy, and three Fog checkpoints support inference without fetching model weights.

From the repository root, create an environment and install the dependencies:

```bash
conda create -n dip-cn python=3.10 -y
conda activate dip-cn
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r DIP/training/requirements.txt -r model_share/requirements.txt
```

### Prepare the dataset locally

Download the [official WBCBench 2026 release](https://www.kaggle.com/competitions/wbc-bench-2026) through the organizers. Extract your authorized archive outside the repository, then arrange or link the JPEGs under `DATA/` in the layout below. Use the original eight-digit image IDs as filenames, including leading zeros. Both evaluation commands below use the checked-in manifest for split membership, labels, and quality features.

```bash
unzip /path/to/wbc-bench-2026.zip -d /path/to/wbcbench-extracted
mkdir -p DATA/{pristine,degraded}
```

```text
DATA/
├── pristine/
│   ├── phase1_train/
│   ├── phase2_train/
│   ├── phase2_eval/
│   └── phase2_test/
└── degraded/
    ├── phase2_train/
    ├── phase2_eval/
    └── phase2_test/
```

For example, if the archive exposes the seven split directories directly under `pristine/` and `degraded/`, link them as follows:

```bash
WBC_EXTRACTED=/path/to/wbcbench-extracted
for split in phase1_train phase2_train phase2_eval phase2_test; do
  ln -s "$WBC_EXTRACTED/pristine/$split" "DATA/pristine/$split"
done
for split in phase2_train phase2_eval phase2_test; do
  ln -s "$WBC_EXTRACTED/degraded/$split" "DATA/degraded/$split"
done
```

The main models use full images. Evaluation resizes images and normalizes them within the loaders. Cellpose crops are used in the separate segmentation experiment.

### Evaluate the published checkpoints

From the repository root, run the final single model and the top-three ensemble on both official evaluation splits. Results go under the ignored `DIP/training/results/reproduced/` directory.

```bash
checkpoint_dir=DIP/training/checkpoints
best_checkpoint="$checkpoint_dir/effb0_224_bs64_mixed_focal_reran/best.pt"
for split in phase2_eval phase2_test; do
  python DIP/training/scripts/evaluate.py \
    --data-root DATA --split "$split" --folder pristine,degraded \
    --checkpoint "$best_checkpoint" \
    --out-dir "DIP/training/results/reproduced/single/$split"
  python DIP/training/scripts/ensemble_eval.py \
    --data-root DATA --split "$split" --folder pristine,degraded --tta full \
    --checkpoints "$best_checkpoint" \
      "$checkpoint_dir/effb0_224_bs64_mixed_eff/best.pt" \
      "$checkpoint_dir/effb0_224_bs64_mixed/best.pt" \
    --out-dir "DIP/training/results/reproduced/ensemble/$split"
done
```

To retrain the focal-loss model using its recorded configuration, run the following from the repository root:

```bash
(
  cd DIP/training
  python scripts/train.py \
    --config results/effb0_224_bs64_mixed_focal_reran/config.yaml \
    --data-root ../../DATA --run-name reproduced/effb0_focal
)
```

### Validate TCP and run the tests

Run these commands from the repository root. The TCP validator uses the shipped ONNX model and Fog ensemble. The unit and integration tests use test fixtures.

```bash
PYTHONPATH=CN/src python -m cn.edge.validate_tcp \
  --images 5 --fog-tta full --output-dir DIP/training/results/reproduced/tcp
PYTHONPATH=DIP/training:CN/src python -m pytest -q DIP/training/tests CN/tests
```

To recompute the policy comparison, run the following in a working copy with
the local dataset linked. It regenerates the summary and figures under
`CN/results/cn_edge_fog/`.

```bash
PYTHONPATH=CN/src python -m cn.edge.run_policy_experiments --fog-tta full
```

### Run the Streamlit demo

Start the app from the repository root:

```bash
python -m streamlit run DIP/training/app.py
```

The app shows a heuristic estimate of image appearance and degradation severity.
When an uploaded filename matches a dataset image ID, it also shows the manifest
class and severity. If the learned policy selects Edge, select **Run Fog prediction**
to compare it with the Fog ensemble on the same image. Demo images are kept
locally. The footer shows the Git
commit, full Edge model SHA-256, and ONNX Runtime and Pillow versions.

---

## Data and licence

The dataset is WBCBench 2026, available at
[`Xin-Tian/wbcbench2026`](https://huggingface.co/datasets/Xin-Tian/wbcbench2026).
Dataset-derived images must not be redistributed. This repository contains only
code, manifests, model checkpoints, and evaluation outputs.

## Known limitations

- Rare classes PC, PLY, and BNE dominate the remaining classification error.
- The INT8 Edge model trades macro-F1 for lower latency.
- The five-fold cross-validation is image-level, not patient-safe.
- Cellpose segmentation did not improve classification and is not integrated into the deployed path.
- Device/phone latency benchmarking has not been recorded.
