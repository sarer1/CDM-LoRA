# CDM-LoRA

Code for **CDM-LoRA: Modeling Cross-Modal Consensus and Discrepancy for Remote Sensing Segmentation**. CDM-LoRA uses paired features to construct low-rank updates inside a frozen DINOv3 encoder. Relational Exchange (REX) models feature consensus and discrepancy; Shared Class Readout (SCR) supplies class feedback and modulates the relational updates.

## Contents

- `source/`: DINOv3 model, training loop, and dataset loaders used by the controlled experiments.
- `run_experiments.py` and `evaluate.py`: controlled training and tile-wise evaluation.
- `configs/`: main CDM-LoRA (DINOv3 ViT-L/16) configurations for Vaihingen, Potsdam and WHU-OPT-SAR.
- `source_manifest.json`: hashes for verifying the retained sources in this checkout.

## Setup

Use Linux with a CUDA-enabled PyTorch installation and one available GPU. Install the remaining dependencies with `python -m pip install -r requirements.txt`. Install the official [DINOv3 repository](https://github.com/facebookresearch/dinov3) at the location expected by `source/MedSAM/models/ImageEncoder/vit/mm_dinov3.py` (`/root/autodl-tmp/dinov3` in the archived layout).

Obtain [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) or WHU-OPT-SAR from the dataset providers. Match the paths in `source/utils.py`: `/root/autodl-fs/` for ISPRS and `/root/autodl-tmp/whu-opt-sar/` for WHU-OPT-SAR. The WHU-OPT-SAR split is in `source/division.txt`. Use symbolic links if your data are stored elsewhere so the source file hashes still match.

Place the official `dinov3_vitl16_pretrain_lvd1689m_8aa4cbdd.pth` in `weights/` under the assets directory passed to the runner.

## Train and evaluate the main model

```sh
python run_experiments.py train --dataset Vaihingen \
  --legacy-root /path/to/assets --output-root ./runs
```

Use `--dataset Potsdam` or `--dataset WHU-OPT-SAR` to run the same model on the other datasets. The runner writes a run configuration under the output directory. After training, the run summary records the selected checkpoint.

Evaluate a trained run at stride 32 with the config produced for that run:

```sh
python evaluate.py --config /path/to/run/run_config.json \
  --checkpoint /path/to/selected.pth \
  --pretrained /path/to/assets/weights/dinov3_vitl16_pretrain_lvd1689m_8aa4cbdd.pth \
  --output ./outputs/evaluation
```

The checked-in configurations use this release's source fingerprints. When evaluating an existing checkpoint with its selection metadata, use the original run configuration so the checkpoint and configuration can be verified together.

## Upstream code

The implementation builds on [SSRS/MFNet](https://github.com/sstary/SSRS), with inherited MedSAM and LoRA components. See [third-party notices](THIRD_PARTY_NOTICES.md) and the preserved source headers. DINOv3 is an external dependency.
