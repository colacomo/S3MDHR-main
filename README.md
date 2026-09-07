# S3MDHR: Sparse Spatial-Spectral Priors with Model-Driven Network for All-in-One Hyperspectral Image Restoration

**Official implementation of **S3MDHR** — a model-driven restoration network that
handles *all-in-one* hyperspectral image (HSI) restoration.** 



## Requirements

- Python 3.8+ (developed and verified with 3.8)
- PyTorch (CUDA build) and a CUDA-capable GPU (≥ 12 GB memory recommended for
  256×256 inference)

Main Python packages:

```
numpy einops timm tifffile pywt tqdm tensorboardX opencv-python
pandas matplotlib
openai-clip        # imported by train.py / test.py / utils/dataset.py
```


## Repository Structure

```
S3MDHR-main
├── options.py                     # global argparse options
├── train.py                       # training + periodic validation + final test
├── test.py                        # per-degradation-type evaluation & visualization
├── models/
│   ├── SMDHR.py                   # main network: SMDHR / SMDHR_b
│   └── SMDHR_modules.py           # FRMoE: experts, AdapterLayer, RoutingFunction
│   
├── utils/
│   ├── dataset.py                 # S3MDHRDataset (train/val loader) & TestDataset (test)
│   ├── lossfunc.py                # HyperspectralSWTLoss, SAMLoss, BandWiseMSE
│   ├── metrics.py                 # PSNR, SAM, RMSE, ERGAS
│   ├── DRCT.py                    # legacy Swin-style backbone (not used by SMDHR_b)
│   ├── modules.py                 # legacy prompt/attention modules (unused copy)
│   ├── glovegenerator.py          # legacy GloVe embedding helper (unused)
│   └── utils_word_embedding.py    # legacy word-embedding helpers (unused)
├── data_generation/
│   └── hyperspectral_processing_modified.m   # MATLAB: synthesize 15 degradations
└── ops/                           # custom CUDA ops (DCN / fused act / upfirdn2d;
                                   #   legacy — not required by SMDHR_b)
```

## Dataset Preparation

The default data layout expected by `S3MDHRDataset` / `TestDataset` is:

```
<root>/
├── train/
│   ├── clean/                     # ground-truth TIFFs, e.g. xxxx.tif
│   └── degraded/
│       ├── h/                     # haze
│       ├── b/                     # blur
│       ├── n/                     # noise
│       ├── bm/                    # band missing
│       └── h_b/ ... h_b_n_bm/     # composite degradations (15 types total)
└── test/
    ├── clean/
    └── degraded/...
```


- Set `--root` in `options.py` to your dataset path.

### Degradation Synthesis (Optional)

`data_generation/hyperspectral_processing_modified.m` is a MATLAB script that
generates the 15 degradation types (all binary combinations of haze / blur /
noise / band-missing) from clean images. Edit the input/output paths inside the
script before running.

## Training

1. Prepare the dataset as described above.
2. Set hyper-parameters / paths in `options.py` (model name, data root,
   batch size, learning rate, epochs, pretrained weights for resume, etc.).
3. Launch training:

```bash
python train.py
```

## Testing

Run inference on all 15 degradation types and produce metrics + spectral-diff
visualizations:

```bash
python test.py
```



## Citation

If you find this work useful, please consider citing the paper:

```
S3MDHR: Sparse Spatial-Spectral Priors with Model-Driven Network
for All-in-One Hyperspectral Image Restoration
```

## Acknowledgement

This repository builds upon ideas/code from prior all-in-one HSI restoration
works (e.g. PromptHSI-style prompt-conditioned modules and Restormer-style
transformer utilities); parts of `ops/` come from the original DCN authors.

