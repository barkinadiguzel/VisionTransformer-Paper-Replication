# 🖼️ Vision Transformer (ViT) From Scratch — *Replicating “An Image is Worth 16x16 Words”*

Reimplementation of the **Vision Transformer (ViT) architecture** proposed in  
📄 [Dosovitskiy et al., 2020 — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)

This project reproduces the ViT model entirely **from scratch using PyTorch**.  
Every component — from **patch embedding** and **class token**, **positional encoding**, **multi-head self-attention**, **feed-forward layers**, to the **encoder stack** — follows the original paper and equations, with formulas visually mapped in [`images/summary.png`](images/summary.png).

---

## Model Flow Summary

![ViT Summary](images/summary.png)

- This summary visual matches the ViT architecture with its core formulas. Each step—from patch embedding to the final class token—is linked to the corresponding section in the original paper.

---

## 🧩 Project Structure
```bash

VisionTransformer-Paper-Replicating/
│
├── src/
│ ├── input_embedding/
│ │ ├── patch_embedding.py         → Patch → Linear Embedding (makale: Section 3.2)
│ │ └── positional_encoding.py     → Sinusoidal or Learnable (makale: Section 3.2)
│ │
│ ├── attention/
│ │ ├── scaled_dot_product.py      → softmax(QKᵀ / √dₖ)V (makale: Section 3.2.1)
│ │ └── multi_head_attention.py    → Concat(head₁,…,headₕ)W₀ (makale: Section 3.2.2)
│ │
│ ├── feed_forward/
│ │ └── positionwise_ffn.py        → FFN(x)=max(0,xW₁+b₁)W₂+b₂ (makale: Section 3.3)
│ │
│ ├── encoder/
│ │ ├── encoder_layer.py           → Attention + FFN + Residual + LayerNorm (makale: Section 3.1)
│ │ └── encoder_stack.py           → N-layer encoder stack (makale: Section 3.1)
│ │
│ ├── vit_model/
│ │ └── vit_assembly.py            → Patch Embedding + Encoder + Class token → final ViT (makale: Section 3)
│ │
│
├── images/
│ └── summary.png
│
└── requirements.txt

```
---
## 🔗 Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)


