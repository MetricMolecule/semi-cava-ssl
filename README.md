# SEMI-CAVA: Semi-Supervised Causal Variational Autoencoder

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

---

## 🚀 Running the Model

```bash
python src.training.train
```
---

## 📷 Medical Dataset Samples

The model was also evaluated on MedMNIST datasets.

### BloodMNIST
![BloodMNIST](assets/medical_samples/bloodmnist.png)

### BreastMNIST
![BreastMNIST](assets/medical_samples/breastmnist.png)

### PneumoniaMNIST
![PneumoniaMNIST](assets/medical_samples/pneumoniamnist.png)

### OrganMNIST (Chest)
![OrganMNIST](assets/medical_samples/organ_c_mnist.png)

---

## 🔗 License

MIT License

---

## Citation

```bibtex
@ARTICLE{11105572,
author={Saha, Saptarshi and Sahoo, Pratyush Kumar and Garain, Utpal},
journal={ IEEE Transactions on Pattern Analysis \& Machine Intelligence },
title={{ SEMI-CAVA: A Causal Variational Approach to Semi-Supervised Learning }},
year={2025},
volume={47},
number={11},
ISSN={1939-3539},
pages={10022-10032},
abstract={ Deep learning has advanced rapidly, but relies heavily on large-labeled datasets for effective training. This is particularly challenging in fields like medicine, where expert labeling is costly, labor-intensive, and prone to bias and error. Semi-supervised learning (SSL) addresses this challenge by reducing reliance on labeled data. SSL is closely tied to the concept of causation. However, recent works relating causality to SSL are limited by modeling only low-dimensional observations or designing a plug-in module to alleviate the class imbalance. In this paper, we take steps towards training causal generative models for semi-supervised learning, combining principles from causality and variational inference. We interpret the Mixup strategy as a stochastic intervention and introduce a consistency loss to promote coherent latent representations. Under reasonable assumptions, we provide theoretical guarantees that the learned latent representations align with true causal factors up to permissible ambiguities. The experimental results show the proposed approach achieves state-of-the-art performance on several medical datasets of different modalities. Additionally, we test our model on standard benchmarking datasets: CIFAR10, CIFAR100, and SVHN, where it achieves competitive performance. },
keywords={Semisupervised learning;Training;Cause effect analysis;Stochastic processes;Data mining;Benchmark testing;Autoencoders;Accuracy;Standards;Random variables},
doi={10.1109/TPAMI.2025.3594360},
url = {https://doi.ieeecomputersociety.org/10.1109/TPAMI.2025.3594360},
publisher={IEEE Computer Society},
address={Los Alamitos, CA, USA},
month=nov}
```


## 📧 Contact

GitHub: [MetricMolecule](https://github.com/MetricMolecule) \
Email Address: [✉️](mailto:[anshak001@gmail.com])

Feel free to reach out for any queries, suggestions, or collaborations. It would be a pleasure to hear from you.

---
