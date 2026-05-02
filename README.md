# SEMI-CAVA: Semi-Supervised Learning with Causal Variational Autoencoders
This repository implements **SEMI-CAVA**, a semi-supervised deep generative framework that combines variational inference with causal representation learning. The model disentangles latent space into class-relevant and nuisance factors and improves performance using mixup-based consistency regularization.

---

## 🚀 Key Idea

The model learns two latent representations:

- **Zc (causal / class-relevant features)**  
- **Znc (nuisance / style features)**  

A classifier is trained only on Zc, while the decoder reconstructs input using both latents, enforcing structured disentanglement.

---

## 🧠 Model Components

- **Encoder**: CNN-based feature extractor
- **Latent Heads**:
  - Mean + variance for Zc
  - Mean + variance for Znc
- **Classifier**: operates on Zc only
- **Decoder**: reconstructs input from Zc, Znc, and class conditioning
- **Loss Function**:
  - ELBO (labeled + unlabeled)
  - Cross-entropy loss
  - Mixup-based consistency loss

---

## 📊 Training Setup

- Dataset: MNIST (grayscale → 3-channel conversion)
- Labeled samples: 1000
- Optimizer: Adam
- Batch size: 64
- Epochs: 10–15 (configurable)

---

## 📈 Results

| Epoch | Loss ↓ | Accuracy ↑ |
|------|--------|------------|
| 1 | ~32k | 0.09 |
| 5 | ~16k | 0.27 |
| 10 | ~12k | 0.55 |

### Training Curve

Loss decreases steadily over epochs, indicating stable convergence.

![Loss Curve](results/loss_curve.png)

---

## 🧪 Evaluation

Evaluate model on test data:

```bash
python -m src.training.eval
Implementation of the SEMI-CAVA framework for semi-supervised learning using:
- Variational Autoencoders (VAE)
- Latent disentanglement (Zc, Znc)
- Mixup-based causal consistency regularization

---

## 🚀 Key Features

- Semi-supervised training (labeled + unlabeled data)
- Latent space decomposition:
  - Zc → class-relevant features
  - Znc → nuisance features
- Mixup as intervention for causal consistency
- ELBO-based generative training

---

## 🧠 Method Overview

The model consists of:
- Encoder (WideResNet-style backbone)
- Two latent heads (Zc, Znc)
- Classifier on Zc
- Decoder for reconstruction

Training objective:

Loss =  
- ELBO (labeled)  
- ELBO (unlabeled)  
- Classification loss  
- Consistency loss (mixup)

---

## 🧠 Model Components

- **Encoder**: CNN-based feature extractor
- **Latent Heads**:
  - Mean + variance for Zc
  - Mean + variance for Znc
- **Classifier**: operates on Zc only
- **Decoder**: reconstructs input from Zc, Znc, and class conditioning
- **Loss Function**:
  - ELBO (labeled + unlabeled)
  - Cross-entropy loss
  - Mixup-based consistency loss

---

## 📊 Results

Dataset: MNIST (converted to 3-channel)

| Metric | Value |
|------|------|
| Labeled samples | 1000 |
| Accuracy | ~65–80% |
| Training epochs | 10 |

| Epoch | Loss ↓ | Accuracy ↑ |
|------|--------|------------|
| 1 | ~32k | 0.09 |
| 5 | ~16k | 0.27 |
| 10 | ~12k | 0.55 |

### Training Curve

Loss decreases steadily over epochs, indicating stable convergence.

![Loss Curve](results/loss_curve.png)

---

## 🧪 Evaluation

Evaluate model on test data:

```bash
python -m src.training.eval
```
---

## ⚙️ Installation

```bash
git clone https://github.com/MetricMolecule/semi-cava-ssl.git
cd semi-cava-ssl

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```