<div align="center">

# 🧬 SEMI-CAVA 
**A Causal Variational Approach to Semi-Supervised Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![IEEE TPAMI](https://img.shields.io/badge/Paper-IEEE_TPAMI_2025-b31b1b.svg)](https://doi.ieeecomputersociety.org/10.1109/TPAMI.2025.3594360)

*An advanced semi-supervised deep generative framework combining variational inference with causal representation learning.*

</div>

---

## 🚀 Key Idea

**SEMI-CAVA** is designed to overcome the heavy reliance on large-labeled datasets in deep learning. By treating the **Mixup strategy** as a stochastic intervention and introducing a consistency loss, SEMI-CAVA promotes coherent latent representations that align with true causal factors.

The model learns two structured, disentangled latent representations:
- **$Z_c$**: Causal / class-relevant features
- **$Z_{nc}$**: Nuisance / style features

A classifier is trained exclusively on $Z_c$, while the decoder reconstructs the input using both $Z_c$ and $Z_{nc}$, alongside class conditioning. This enforces structured disentanglement, significantly improving performance—especially in critical domains like medical imaging where expert labeling is costly and sparse.

---

## 🧠 Architecture Overview

- **Encoder**: CNN-based feature extractor mapping inputs to a shared representation.
- **Latent Heads**:
  - Independent branches predicting mean ($\mu$) and variance ($\sigma^2$) for both $Z_c$ and $Z_{nc}$.
- **Classifier**: Operates solely on the causal latent space $Z_c$.
- **Decoder**: Reconstructs the original input from the concatenated representations ($Z_c$, $Z_{nc}$) and the class label.
- **Objective Function**:
  - **ELBO Loss**: Applied to both labeled and unlabeled data for generative reconstruction.
  - **Cross-Entropy Loss**: For supervised classification.
  - **Consistency Loss**: Mixup-based regularization to enforce causal invariance.

---

## 📊 Training Setup & Performance

### ⚙️ Default Configuration
- **Dataset**: MNIST (grayscale → 3-channel conversion for broader compatibility)
- **Labeled samples**: 1000
- **Optimizer**: Adam
- **Batch size**: 64
- **Epochs**: 10–15 (configurable)

### 📈 Results (MNIST Benchmark)

| Epoch | Total Loss ↓ | Accuracy ↑ |
|:---:|:---:|:---:|
| 1 | ~32k | 0.09 |
| 5 | ~16k | 0.27 |
| 10 | ~12k | **0.55** |

<div align="center">
  <img src="results/loss_curve.png" alt="Training Loss Curve" width="600"/>
  <p><em>Stable convergence of training loss over epochs.</em></p>
</div>

---

## 📷 Medical Dataset Evaluation

SEMI-CAVA achieves state-of-the-art performance on several critical medical datasets from the MedMNIST benchmark collection, demonstrating its robustness in real-world, data-scarce domains.

| BloodMNIST | BreastMNIST |
|:---:|:---:|
| <img src="assets/medical_samples/bloodmnist.png" width="250"/> | <img src="assets/medical_samples/breastmnist.png" width="250"/> |

| PneumoniaMNIST | OrganMNIST (Chest) |
|:---:|:---:|
| <img src="assets/medical_samples/pneumoniamnist.png" width="250"/> | <img src="assets/medical_samples/organ_c_mnist.png" width="250"/> |

---

## 💻 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/MetricMolecule/semi-cava-ssl.git
cd semi-cava-ssl

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Running the Model

To train the model from scratch on the default dataset:

```bash
python -m src.training.train
```

### 🧪 Evaluation

To evaluate a trained model on the test dataset:

```bash
python -m src.training.eval 
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you find this code or research helpful in your work, please consider citing the paper:

```bibtex
@ARTICLE{11105572,
  author={Saha, Saptarshi and Sahoo, Pratyush Kumar and Garain, Utpal},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  title={{SEMI-CAVA: A Causal Variational Approach to Semi-Supervised Learning}},
  year={2025},
  volume={47},
  number={11},
  ISSN={1939-3539},
  pages={10022-10032},
  doi={10.1109/TPAMI.2025.3594360},
  publisher={IEEE Computer Society}
}
```

---

## 📧 Contact

**MetricMolecule**  
- **GitHub:** [@MetricMolecule](https://github.com/MetricMolecule)  
- **Email:** [✉️](mailto:anshak001@gmail.com)

Feel free to reach out for any queries, suggestions, or potential collaborations!
