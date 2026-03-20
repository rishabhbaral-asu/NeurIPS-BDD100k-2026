# Bridging the Domain Gap in Autonomous Driving: Evaluating Edge-Optimized VLMs

**[Double-Blind Submission - NeurIPS]**

This repository contains the official PyTorch implementation, training scripts, and evaluation pipelines for the paper: *"Bridging the Domain Gap in Autonomous Driving: Evaluating Edge-Optimized Vision-Language Models for Scene Understanding."*

## 📖 Overview
We provide the codebase to reproduce our custom ~50M parameter Vision-Language Model (VLM), designed to audit the "Geographic Infrastructure Bias" inherent in autonomous driving datasets. Our architecture combines a ConvNeXt-Tiny backbone, a 49-token spatial patch projection, and a custom causal language decoder (MiniLLM), optimized via Homoscedastic Uncertainty weighting for multi-task learning (Scene, Time of Day, Weather).

## 📦 Model Weights and Dataset Disclaimer
**Note for Reviewers:** To maintain strict double-blind anonymity and adhere to anonymous repository storage limits, the pre-trained `.pth` weights for our custom VLM, the Ollama baseline outputs, and the high-resolution video files for the OOD dataset are not hosted directly in this repository. 

The complete training scripts, architectural definitions, and evaluation pipelines are provided here for full methodological transparency. Upon publication, all model checkpoints, the 1,000-frame curated OOD dataset (Tokyo, Oslo, Bay Area), and the full training corpus will be publicly released on HuggingFace and Zenodo.

## 📁 Repository Structure
```text
├── data/
│   ├── sample_id/                # Sample images from BDD100K (In-Distribution)
│   └── sample_ood/               # Sample curated images from Tokyo/Oslo (Out-of-Distribution)
├── vlm.py                        # Core VLM architecture (ConvNeXt + Spatial Projector)
├── minigpt.py                    # Causal language decoder architecture (MiniLLM)
├── build_ood.py                  # Algorithmic curation (Laplacian variance, yt-dlp, ego-vehicle cropping)
├── train_model.py                # Main training loop (supports multi-seed runs & uncertainty loss)
├── VLM_agent.py                  # Evaluation script for ID and OOD testing (Semantic Exact Match)
├── ollama_baseline_eval.py       # Wrapper for Moondream2 / Ollama baseline inference
├── requirements.txt              # Environment dependencies
└── README.md
```
## 🚀 Quickstart

### 1. Installation
Ensure you have Python 3.9+ installed. We recommend using a virtual environment (e.g., Conda).

```bash
git clone <anonymous-repo-url>
cd vlm-domain-gap
pip install -r requirements.txt
```
### 2. Dataset Curation
To view the algorithmic filtering pipeline used to curate high-variance, open-vocabulary driving footage from raw video sources, see the logic within build_ood.py:
```bash
python build_ood.py
```
### 3. Training the Custom VLM
To train the model from scratch on the BDD100K dataset, run the following command. The script inherently supports calculating statistical variance across multiple seeds.
```bash
python train_model.py
```
### 4. Evaluation (In-Distribution vs. Out-of-Distribution)
To evaluate the trained model's Semantic Exact Match on the ID (BDD100K) and OOD (Global Curated) datasets:
```bash
python VLM_agent.py
```
### 5. Reproducing Baseline Metrics
To reproduce the zero-shot baseline metrics via Ollama, ensure the Ollama daemon is running locally and execute:
```bash
python ollama_baseline_eval.py
```
# 🔬 Core Contributions Validated in this Codebase
## Multi-Task Uncertainty Loss: Implemented within train_model.py for the dynamic weighting of Weather, Time, and Scene tasks.

## Spatial Geometry Preservation: See vlm.py for the 49-token spatial projection bridging the vision encoder and language decoder.

## Automated Curation: See build_ood.py for the robust filtering pipeline deployed prior to Human-in-the-Loop verification.
