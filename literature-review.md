# Literature Review: Vision Mamba (ViM) & Selective State Space Models for Visual Representation Learning

## Summary
Vision Mamba (ViM) introduces a pure Selective State Space Model (SSM) backbone for computer vision, addressing two major limits of standard 1D Mamba: unidirectionality and lack of spatial location awareness. By converting 2D images into flattened patch tokens, embedding position vectors, and processing sequences through bidirectional 1D selective scan blocks with hardware-aware SRAM/HBM memory scheduling, ViM achieves linear time complexity $O(M)$ and memory complexity $O(M)$ with respect to sequence length $M$. Compared to Vision Transformers (ViT/DeiT), ViM yields $2.8\times$ faster inference speed and saves $86.8\%$ GPU memory on high-resolution ($1248 \times 1248$) images.

## Key Findings by Facet

### 1. Vision Transformers vs. Selective State Space Models (SSMs)
- **Vision Transformers (ViT / DeiT)**: Rely on full self-attention with quadratic complexity $O(M^2 D)$. While effective for capturing global context, memory consumption scales quadratically, making high-resolution downstream tasks (detection, segmentation, video action recognition) computationally prohibitive without local windowing tricks (e.g., Swin).
- **Mamba (Gu & Dao, 2023)**: Introduces time-varying parameters $(\mathbf{B}, \mathbf{C}, \Delta)$ into S4, enabling data-dependent selection with linear complexity $O(M DN)$. However, standard Mamba is 1D causal (unidirectional), making it suboptimal for non-causal 2D spatial image structures.

### 2. Bidirectional SSM Architecture in Vision Mamba (ViM)
- **Patch Projection & Position Embedding**: An input image $\mathbf{t} \in \mathbb{R}^{H \times W \times C}$ is projected into patch tokens $\mathbf{x}_p \in \mathbb{R}^{J \times (P^2 C)}$, concatenated with a learnable `[CLS]` token $t_{cls}$, and added with 1D position embeddings $\mathbf{E}_{pos}$.
- **Bidirectional Scanning Algorithm**: Each ViM block splits tokens into forward and backward directions, passes each through 1D Convolution + SiLU + Selective SSM, applies gating via $\operatorname{SiLU}(z)$, and projects back to hidden dimension $D$.
- **Computational FLOPs**: Self-attention consumes $4MD^2 + 2M^2D$ FLOPs, whereas ViM SSM consumes $6MDN + 2MDN = 8MDN$ FLOPs ($N=16$), maintaining strict linear scaling with respect to image token count $M$.

### 3. Comparison with Related Vision SSM Architectures
- **VMamba (SS2D)**: Processes visual tokens via 4-directional scanning (Cross-Scan Module), maintaining a hierarchical architecture (Swin-like pyramid).
- **Vim (Vision Mamba)**: Uses a columnar (plain) architecture similar to ViT/DeiT with 2-directional (forward + backward) scanning, making it directly compatible with self-supervised pretraining (MAE/Masked Autoencoders) and multimodal sequence frameworks.

## Identified Gaps & Opportunities
- **1D Flattener Information Loss**: Flattening 2D spatial patches into a 1D sequence breaks contiguous 2D spatial adjacency (top-bottom neighbor connectivity).
- **Edge Deployment for Video/Action Recognition**: Applying ViM to spatiotemporal video tokens $(T \times H \times W)$ offers massive memory savings, but hardware-aware kernel support on edge NPU/TensorRT environments requires specialized ONNX export strategies.

## References
```bibtex
@inproceedings{zhu2024visionmamba,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Wang, Biao and Xiang, Yihong and Fu, Li and Chen, Yutong and Wang, Sheng and Zhao, Peng and Zhang, Lei and Liu, Yunchao},
  booktitle={ICML},
  year={2024}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
