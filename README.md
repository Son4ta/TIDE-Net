# Carbon-ECS: A Benchmark and a Physics-Decoupled Model for the East China Sea Carbon Flux

<div align="center">
  <img src="https://placehold.co/800x300?text=TIDE-Net+Framework+Diagram" alt="TIDE-Net Framework">
</div>

<p align="center">
  <a href="https://github.com/son4ta/TIDE-Net/stargazers"><img src="https://img.shields.io/github/stars/son4ta/TIDE-Net?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/son4ta/TIDE-Net/network/members"><img src="https://img.shields.io/github/forks/son4ta/TIDE-Net?style=social" alt="GitHub Forks"></a>
  <a href="https://huggingface.co/spaces/[YOUR_HF_SPACE]"><img src="https://img.shields.io/badge/🤗-Hugging Face Demo-blue.svg" alt="Hugging Face Demo"></a>
  <a href="[LINK_TO_PAPER]"><img src="https://img.shields.io/badge/📄-Paper-red.svg" alt="Paper"></a>
</p>

This is the official implementation of **"Carbon-ECS: A Benchmark and a Physics-Decoupled Model for the East China Sea Carbon Flux"**. This work introduces TIDE-Net, a novel, physics-inspired forecasting model, and Carbon-ECS, a high-resolution, spatio-temporally complete benchmark dataset for sea-air CO₂ flux ($F_{CO_2}$) forecasting.

## 🌟 Introduction

Forecasting sea-air CO₂ flux ($F_{CO_2}$) in coastal ecosystems like the East China Sea (ECS) is notoriously difficult due to complex biogeochemical processes and data limitations. TIDE-Net addresses this by decoupling two critical physical phenomena:
1.  **Macro-scale Transport**: The large-scale movement of carbon sinks driven by ocean currents.
2.  **Micro-scale Evolution**: The local generation, diffusion, and dissipation of fine-grained carbon structures.

Our model leverages a dual-branch architecture to model these processes separately, leading to more physically consistent and accurate forecasts.

## ✨ Key Features

* **Carbon-ECS Dataset**: We provide the first 17-year, high-resolution (~1 km), and spatio-temporally complete benchmark dataset for $F_{CO_2}$ forecasting in the ECS.
* **Physics-Decoupled Architecture**: TIDE-Net explicitly separates large-scale transport from local detail evolution for improved physical interpretability and performance.
* **State-of-the-Art Performance**: Our model significantly outperforms existing methods in both short-term and medium-term forecasting tasks.

## 🏗️ Model Architecture

TIDE-Net is composed of three synergistic modules:

1.  **Frequency Global Module (FGM)**: Predicts large-scale dynamics by operating in the frequency domain, modeling the evolution of amplitude and phase separately.
2.  **Diffusion Local Refiner (DLR)**: Generates high-fidelity local details by iteratively refining fine-grained spatial structures using a diffusion-based process.
3.  **Synergistic Fusion Head (SFH)**: Adaptively fuses the global forecast from FGM and the local details from DLR to produce the final, refined prediction.

<div align="center">
  <img src="https://placehold.co/700x400?text=Detailed+Architecture+Diagram" alt="TIDE-Net Architecture">
</div>

## 💾 Carbon-ECS Benchmark Dataset

To overcome the limitations of spatio-temporal discontinuity in satellite data, we developed the Carbon-ECS dataset.

| Specification       | Value                                        |
| ------------------- | -------------------------------------------- |
| Temporal Coverage   | 2003–2019 (204 months)                       |
| Temporal Resolution | Monthly                                      |
| Spatial Coverage    | Yangtze River Estuary & ECS shelf            |
| Spatial Resolution  | ~1 km ($0.01^\circ$)                         |
| Area                | ~8.5 $\times$ 10$^5$ km$^2$                  |
| Data Quality        | Complete, denoised, and bias-corrected       |

The dataset and its generation pipeline are available at [https://github.com/Son4ta/TIDE-Net](https://github.com/Son4ta/TIDE-Net).

## 🚀 Getting Started

### 1. Environment Setup

We recommend using Conda to manage the environment.

```bash
# Clone the repository
git clone [https://github.com/Son4ta/TIDE-Net.git](https://github.com/Son4ta/TIDE-Net.git)
cd TIDE-Net

# Create and activate the conda environment
conda env create -f env.yaml
conda activate tide-net


Our code utilizes `accelerate` from Hugging Face for multi-GPU training and inference. You can configure it easily:

```bash
accelerate config
```

### 2\. Download Pretrained Models

Download the pretrained model checkpoints and place them in the `./resources/` directory.

| Model                                | Download Link                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------- |
| TIDE-Net on Carbon-ECS (Short-Term)  | [Google Drive Placeholder](https://www.google.com/search?q=https://drive.google.com/file/d/%5BYOUR_FILE_ID%5D/view) |
| TIDE-Net on Carbon-ECS (Medium-Term) | [Google Drive Placeholder](https://www.google.com/search?q=https://drive.google.com/file/d/%5BYOUR_FILE_ID%5D/view) |

## ⚙️ Usage

### Training

To train TIDE-Net from scratch, run the following command. Make sure to configure the dataset path in `dataset/get_datasets.py`.

```bash
python run.py --exp_note "my_first_training_run"
```

You can view all available training options and hyperparameters by running:

```bash
python run.py -h
```

### Evaluation

To evaluate a pretrained model, use the `--eval` flag and specify the path to the checkpoint.

```bash
# Example for evaluating the medium-term model
python run.py --eval --ckpt_milestone ./resources/[CHECKPOINT_NAME].pt
```

## 📊 Results

TIDE-Net sets a new state-of-the-art on the Carbon-ECS benchmark. It consistently outperforms leading spatio-temporal forecasting models across multiple metrics.

\<div align="center"\>
\<img src="https://www.google.com/search?q=https://placehold.co/800x400%3Ftext%3DQualitative%2BResults%2BComparison" alt="Qualitative Results"\>
\<p\>\<em\>Qualitative comparison of 6-month-ahead forecasts. TIDE-Net's predictions are visually more consistent with the ground truth.\</em\>\</p\>
\</div\>

## 🙏 Acknowledgements

This project's foundational training and evaluation framework is adapted from [AlphaPre](https://github.com/linkenghong/AlphaPre). We are deeply grateful for their excellent open-source contribution to the community.

We also thank the authors of the following repositories for their inspiration and for providing valuable codebases:

  - [OpenSTL](https://github.com/chengtan9907/OpenSTL)
  - [DiffCast](https://github.com/DeminYu98/DiffCast)

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{fang2025carbon,
  title={Carbon-ECS: A Benchmark and a Physics-Decoupled Model for the East China Sea Carbon Flux},
  author={Fang, Chengjie and Zhao, Enyuan and Yao, Hongming and Sun, Ziyu and Gao, Di and Li, Yanbiao},
  booktitle={Preprint},
  year={2025},
  note={Details to be updated upon publication.}
}
```

```
```
